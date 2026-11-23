/*
 * audioclip native soft clipper (plain C shared library, loaded via ctypes).
 *
 * Signal chain (causal; filter state held inside the handle across calls):
 *   1. gain (drive)
 *   2. 4x oversampling via zero-order hold: every input yields four equal
 *      sub-samples, giving an exact phase-0 unit impulse so a freshly built
 *      clipper reproduces small single samples accurately
 *   3. odd-symmetric, non-decreasing soft saturation on the 4x grid
 *   4. true-peak protection: every oversampled sub-sample is hard bounded to
 *      [-L4, L4] with L4 < 1, so the oversampled true peak is <= L4 < 1.0
 *   5. zero-delay anti-imaging decimation FIR whose taps are non-negative and
 *      monotonically non-increasing and sum to 1 (order preserving, odd,
 *      unity DC); the dominant first tap keeps cold-start error small
 *   6. output safety limiter to [-LOUT, LOUT] (LOUT < 1)
 *
 * Both the saturation and the FIR are non-decreasing odd maps built from
 * non-negative coefficients, therefore the whole operator is non-decreasing
 * and odd. Peaks are bounded on the oversampled grid before decimation, so
 * the post-clip true peak can never exceed 1.0.
 */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define AC_OVERSAMPLE 4
#define AC_NTAP 16
#define AC_LINEAR 0.5          /* linear region of the saturation curve */
#define AC_L4 0.985            /* 4x true-peak ceiling (< 1.0) */
#define AC_LOUT 0.975          /* base-rate output ceiling (< 1.0) */
#define AC_W0 0.93             /* decimator dominant tap */
#define AC_R 0.72              /* geometric tail ratio */

/* distinct, stable status codes so callers can classify failures */
#define AC_OK 0
#define AC_ERR_BAD_DRIVE 1
#define AC_ERR_NONFINITE 2
#define AC_ERR_MEMORY 3

typedef struct {
    double drive;
    double dec[AC_NTAP];   /* decimation FIR coefficients (dec[0] dominant) */
    double hist[AC_NTAP];  /* newest-first 4x sub-sample history */
    int primed;            /* whether history has been DC-seeded yet */
} ac_clip_t;

static void ac_init_coeffs(ac_clip_t *st)
{
    /* geometric, strictly non-increasing tail; normalized so the sum is 1. */
    double tail[AC_NTAP - 1];
    double sum = 0.0;
    for (int k = 0; k < AC_NTAP - 1; ++k) {
        tail[k] = pow(AC_R, k + 1);
        sum += tail[k];
    }
    st->dec[0] = AC_W0;
    for (int k = 0; k < AC_NTAP - 1; ++k)
        st->dec[k + 1] = (1.0 - AC_W0) * tail[k] / sum;
    memset(st->hist, 0, sizeof(st->hist));
}

ac_clip_t *ac_create(double drive)
{
    if (!isfinite(drive) || drive <= 0.0)
        return NULL;
    ac_clip_t *st = (ac_clip_t *)calloc(1, sizeof(ac_clip_t));
    if (st == NULL)
        return NULL;
    st->drive = drive;
    ac_init_coeffs(st);
    return st;
}

void ac_destroy(ac_clip_t *st)
{
    free(st);
}

/* odd, non-decreasing soft saturation; identity for |x| <= AC_LINEAR. */
static inline double ac_curve(double x)
{
    double ax = fabs(x);
    if (ax <= AC_LINEAR)
        return x;
    /* exponential asymptote toward 1, joined C1 at AC_LINEAR. */
    double v = 1.0 - (1.0 - AC_LINEAR) *
        exp(-(ax - AC_LINEAR) / (1.0 - AC_LINEAR));
    if (v > AC_L4)              /* true-peak guard on the oversampled grid */
        v = AC_L4;
    return x < 0.0 ? -v : v;
}

int ac_process(ac_clip_t *st, const double *in, size_t n, double *out)
{
    if (n == 0)
        return AC_OK;
    /* DC-consistent warm start: seed history with the first saturated value
     * so a monotone input of either sign maps to a monotone output and a
     * single sample is reproduced exactly. Empty calls leave state untouched. */
    if (!st->primed) {
        double x0 = in[0];
        if (!isfinite(x0))
            return AC_ERR_NONFINITE;
        double c0 = ac_curve(x0 * st->drive);
        for (int k = 0; k < AC_NTAP; ++k)
            st->hist[k] = c0;
        st->primed = 1;
    }
    for (size_t i = 0; i < n; ++i) {
        double x = in[i];
        if (!isfinite(x))
            return AC_ERR_NONFINITE;
        double g = x * st->drive;
        double y = 0.0;
        for (int p = 0; p < AC_OVERSAMPLE; ++p) {
            /* zero-order hold: all four sub-samples equal the current input */
            double c = ac_curve(g);
            memmove(st->hist + 1, st->hist,
                    (AC_NTAP - 1) * sizeof(double));
            st->hist[0] = c;
            if (p == 0) {
                /* zero-delay decimation aligned to this phase */
                double acc = 0.0;
                for (int k = 0; k < AC_NTAP; ++k)
                    acc += st->dec[k] * st->hist[k];
                if (acc > AC_LOUT)
                    acc = AC_LOUT;
                else if (acc < -AC_LOUT)
                    acc = -AC_LOUT;
                y = acc;
            }
        }
        out[i] = y;
    }
    return AC_OK;
}
