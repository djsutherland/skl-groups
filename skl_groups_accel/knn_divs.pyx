from __future__ import division

cimport cython
from cython cimport view
from cython.parallel import prange, threadid
from libc.stdlib cimport malloc, free
from libc.math cimport log, sqrt, fmax
from cpython.exc cimport PyErr_CheckSignals

from functools import partial

import numpy as np
cimport numpy as np
from numpy cimport uint8_t

from cyflann.flann cimport flann_index_t, FLANNParameters, \
                           flann_find_nearest_neighbors_index_float
from cyflann.index cimport FLANNIndex, FLANNParameters as CyFLANNParameters

from skl_groups.divergences._knn import (_linear as py_linear,
                                         kl as py_kl,
                                         _alpha_div as py_alpha_div,
                                         _jensen_shannon_core as py_js_core)



cdef float fnan = float("NaN")
cdef float finf = float("inf")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _linear(const float[::1] Bs, int dim, int num_q,
                  const float[:, ::1] nus, bint clamp,
                  float[:] results) nogil:
    #   B / m * mean(nu ^ -dim)
    cdef int i, j
    cdef int num_p = nus.shape[0]
    cdef int num_Ks = results.shape[0]
    cdef float mean
    cdef float mdim = -dim

    for j in range(num_Ks):
        mean = 0
        for i in range(num_p):
            mean += (nus[i, j] ** mdim) / num_p
        mean *= Bs[j] / num_q
        if clamp and mean < 0:
            mean = 0
        results[j] = mean


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void kl(int dim, int num_q,
             const float[:, ::1] rhos, const float[:, ::1] nus, bint clamp,
             float[:] results) nogil:
    # dim * mean(log(nus) - log(rhos), axis=0) + log(num_q / (num_p - 1))

    cdef int i, j
    cdef int num_p = rhos.shape[0]
    cdef int num_Ks = results.shape[0]
    cdef float mean

    cdef float cons = log(num_q / (<float> (num_p - 1)))

    for j in range(num_Ks):
        mean = 0
        for i in range(num_p):
            mean += (log(nus[i, j]) - log(rhos[i, j])) / num_p
        mean = dim * mean + cons
        if clamp and mean < 0:
            mean = 0
        results[j] = mean


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _alpha_div(const float[::1] omas, const float[:, ::1] Bs,
                     int dim, int num_q,
                     const float[:, ::1] rhos, const float[:, ::1] nus,
                     bint clamp,
                     const int[::1] poses, float[:, :] results) nogil:
    cdef int i, j, k
    cdef int num_alphas = omas.shape[0]
    cdef int num_p = rhos.shape[0]
    cdef int num_Ks = rhos.shape[1]
    cdef float ratio, factor
    cdef float nump1_q = (<float>(num_p - 1)) / num_q

    for i in range(num_alphas):
        for j in range(num_Ks):
            results[poses[i], j] = 0

    # the actual main estimate:
    #   mean( rho^(- dim * est alpha) nu^(- dim * est beta) )
    #   = mean( (rho / nu) ^ (dim * (1 - alpha)) )
    for k in range(num_p):
        for j in range(num_Ks):
            ratio = rhos[k, j] / nus[k, j]
            for i in range(num_alphas):
                results[poses[i], j] += ratio ** (dim * omas[i]) / num_p

    for i in range(num_alphas):
        factor = nump1_q ** omas[i]
        for j in range(num_Ks):
            results[poses[i], j] *= factor * Bs[i, j]
            if clamp and results[poses[i], j] < 0:
                results[poses[i], j] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _jensen_shannon_core(const int[::1] Ks, int dim,
                               int min_i, const float[::1] digamma_vals,
                               int num_q,
                               const float[:, ::1] rhos,
                               const float[:, ::1] nus,
                               const int[::1] Ks_order, float min_sq_dist,
                               float[:] alphas_tmp, float[:] results) nogil:
    # NOTE: rhos contains all the neighbors up to max_K
    # NOTE: nus here is the "dists_out" array, which is a squared distance
    #       that hasn't been thresholded by min_dist
    cdef int i
    cdef int num_p = rhos.shape[0]

    cdef double t = 2 * num_p - 1
    cdef double p_wt = 1 / t
    cdef double q_wt = num_p / (num_q * t)

    cdef int max_K = rhos.shape[1]
    cdef int num_Ks = Ks.shape[0]

    cdef double alpha, max_wt = -1
    for i in range(num_Ks):
        alphas_tmp[i] = alpha = Ks[i] / (num_p + num_q - 1.)
        if alpha > max_wt:
            max_wt = alpha

    for i in range(num_Ks):
        results[i] = 0

    # mergesort rhos and nus
    # keeping track of the incremental weights until we hit each alpha
    cdef double curr_quantile, log_curr_dist, log_last_dist
    cdef double next_rho_log_dist, next_nu_log_dist
    cdef int next_rho, next_nu, next_alpha

    for i in range(num_p):
        curr_quantile = 0.
        next_alpha = 0
        log_curr_dist = log_last_dist = fnan

        next_rho = 0
        next_rho_log_dist = log(rhos[i, next_rho])

        next_nu = 0
        next_nu_log_dist = log(fmax(min_sq_dist, nus[i, next_nu])) / 2.

        while next_alpha < num_Ks:
            log_last_dist = log_curr_dist
            if next_rho_log_dist < next_nu_log_dist:
                log_curr_dist = next_rho_log_dist
                curr_quantile += p_wt
                next_rho += 1
                if next_rho == max_K:
                    next_rho_log_dist = finf
                else:
                    next_rho_log_dist = log(rhos[i, next_rho])
            else:
                log_curr_dist = next_nu_log_dist
                curr_quantile += q_wt
                next_nu += 1
                if next_nu == max_K:
                    next_nu_log_dist = finf
                else:
                    next_nu_log_dist = \
                        log(fmax(min_sq_dist, nus[i, next_nu])) / 2.

            while (next_alpha < num_Ks and
                   curr_quantile > alphas_tmp[Ks_order[next_alpha]]):
                results[Ks_order[next_alpha]] += (
                    dim * log_last_dist
                    - digamma_vals[next_rho + next_nu - 1 - min_i]
                ) / num_p
                next_alpha += 1


################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _estimate_cross_divs(X_features, X_indices, X_rhos,
                         Y_features, Y_indices, Y_rhos,
                         funcs, the_Ks, int max_K, bint save_all_Ks,
                         int n_output, do_sym, bint to_self,
                         log_progress, int n_jobs, float min_dist, bint clamp):
    
    cdef int a, i, j, k, i_start, i_end, num_i, j_start, j_end, num_j
    cdef int num_p, num_q
    cdef flann_index_t q_idx
    cdef float * p_feats
    cdef bint is_sym

    cdef int[::1] Ks = np.asarray(the_Ks, dtype=np.int32)
    cdef int n_X = len(X_features)
    cdef int n_Y = len(Y_features)
    cdef int n_Ks = Ks.size
    cdef int dim = X_features.dim
    cdef float min_sq_dist = min_dist * min_dist
    cdef bint X_is_Y = X_features == Y_features

    # for now, require everything to be stacked so we can get at it without GIL
    # TODO: figure out how to do this better, given that you can't have
    #       pointers to memoryviews...
    X_features.make_stacked()
    cdef float[:, ::1] X_feats = np.asarray(X_features.stacked_features,
                                            dtype=np.float32)
    cdef long[:] X_boundaries = X_features._boundaries

    Y_features.make_stacked()
    cdef float[:, ::1] Y_feats = np.asarray(Y_features.stacked_features,
                                            dtype=np.float32)
    cdef long[:] Y_boundaries = Y_features._boundaries

    cdef bint do_any_sym = bool(do_sym)
    cdef uint8_t[::1] do_sym_a = np.zeros(n_output, dtype=np.uint8)
    if do_sym == True:
        do_sym_a[:] = True
    else:
        for pos in do_sym:
            if pos < 0:
                pos += n_output
            do_sym_a[pos] = True

    cdef float[:, ::1] X_rhos_stacked, all_X_rhos_stacked
    cdef float[:, ::1] Y_rhos_stacked, all_Y_rhos_stacked

    if save_all_Ks:
        all_X_rhos_stacked = np.ascontiguousarray(
            np.vstack(X_rhos), dtype=np.float32)
        X_rhos_stacked = np.ascontiguousarray(
            np.asarray(all_X_rhos_stacked)[:, np.asarray(Ks) - 1])

        if do_any_sym and not to_self:
            all_Y_rhos_stacked = np.ascontiguousarray(
                np.vstack(Y_rhos), dtype=np.float32)
            Y_rhos_stacked = np.ascontiguousarray(
                np.asarray(all_Y_rhos_stacked)[:, np.asarray(Ks) - 1])
    else:
        X_rhos_stacked = np.ascontiguousarray(
            np.vstack(X_rhos), dtype=np.float32)
        if do_any_sym and not to_self:
            Y_rhos_stacked = np.ascontiguousarray(
                np.vstack(Y_rhos), dtype=np.float32)


    ############################################################################
    ### Figure out which functions we have

    cdef bint do_linear = False
    cdef float[::1] linear_Bs
    cdef int linear_pos

    cdef bint do_kl = False
    cdef int kl_pos

    cdef bint do_alpha = False
    cdef int alpha_num_alphas
    cdef float[::1] alpha_omas
    cdef float[:, ::1] alpha_Bs
    cdef int[::1] alpha_pos

    cdef bint do_js = False
    cdef int js_min_i
    cdef float[::1] js_digamma_vals
    cdef int[::1] js_Ks_order
    cdef int js_pos

    for func, info in funcs.iteritems():
        assert isinstance(func, partial)
        assert func.keywords is None
        real_func = func.func

        if real_func is py_linear:
            assert not do_linear
            do_linear = True
            Bs, the_dim = func.args

            assert Bs.shape == (n_Ks,)
            linear_Bs = np.asarray(Bs, dtype=np.float32)

            assert the_dim == dim

            linear_pos, = info.pos
            if linear_pos < 0:
                linear_pos += n_output

        elif real_func is py_kl:
            assert not do_kl
            do_kl = True
            the_Ks, the_dim = func.args
            assert np.all(the_Ks == Ks)
            assert the_dim == dim

            kl_pos, = info.pos
            if kl_pos < 0:
                kl_pos += n_output

        elif real_func is py_alpha_div:
            assert not do_alpha
            do_alpha = True
            omas, Bs, the_dim = func.args

            alpha_omas = np.asarray(omas.ravel(), dtype=np.float32)
            alpha_num_alphas = alpha_omas.size

            assert Bs.shape == (alpha_num_alphas, n_Ks)
            alpha_Bs = np.asarray(Bs, dtype=np.float32)

            assert the_dim == dim

            alpha_pos = np.asarray(info.pos, dtype=np.int32)
            for i in range(alpha_pos.shape[0]):
                if alpha_pos[i] < 0:
                    alpha_pos[i] += n_output

        elif real_func is py_js_core:
            assert not do_js
            do_js = True
            assert save_all_Ks
            the_Ks, the_dim, js_min_i, the_digamma_vals = func.args
            assert np.all(the_Ks == Ks)
            assert the_dim == dim
            assert the_digamma_vals.ndim == 1
            js_digamma_vals = np.asarray(the_digamma_vals, dtype=np.float32)

            js_Ks_order = np.argsort(Ks).astype(np.int32)

            js_pos, = info.pos
            if js_pos < 0:
                js_pos += n_output

        else:
            msg = "cython code can't handle function {}"
            raise ValueError(msg.format(real_func))

    ############################################################################

    # use passed params, but set cores = 1
    # NOTE: assumes that all indices have same params
    cdef FLANNParameters params = \
        (<CyFLANNParameters> X_indices[0].params)._this
    params.cores = 1

    # the results variable
    cdef float[:, :, :, :, ::1] outputs = np.empty(
        (n_output, n_Ks, n_X, n_Y, 2 if do_any_sym else 1), dtype=np.float32)
    outputs[:, :, :, :, :] = fnan

    # work buffer for each thread; first axis for the thread
    # since cython doesn't currently support thread-local memoryviews :|
    cdef int max_pts = max(np.max(X_features.n_pts), np.max(Y_features.n_pts))

    cdef int[:, :, ::1] idx_out = \
        np.empty((n_jobs, max_pts, max_K), dtype=np.int32)
    cdef float[:, :, ::1] dists_out = \
        np.empty((n_jobs, max_pts, max_K), dtype=np.float32)
    cdef float[:, :, ::1] neighbors = \
        np.empty((n_jobs, max_pts, n_Ks), dtype=np.float32)
    cdef float[:, ::1] alphas_tmp = np.empty((n_jobs, n_Ks), dtype=np.float32)
    cdef int tid
    cdef long job_i
    cdef long n_to_do = n_X * n_Y * (2 if do_any_sym and not to_self else 1)

    cdef object pbar = log_progress
    cdef bint do_progress = bool(pbar)
    cdef long jobs_since_last_tick_val
    cdef long * jobs_since_last_tick = &jobs_since_last_tick_val
    cdef long[::1] num_done  # total things done by each thread
    if do_progress:
        num_done = np.zeros(n_jobs, dtype=np.int64)
        pbar.start(total=n_to_do)

    # make a C array of pointers to indices, to get at it w/o the GIL
    cdef flann_index_t * index_array = <flann_index_t *> malloc(
            (n_X + n_Y) * sizeof(flann_index_t))
    if not index_array:
        raise MemoryError()
    try:
        # populate index_array
        for i in range(n_X):
            index_array[i] = (<FLANNIndex> X_indices[i])._this
        for j in range(n_Y):
            index_array[n_X + j] = (<FLANNIndex> Y_indices[j])._this

        with nogil:
            for job_i in prange(n_to_do, num_threads=n_jobs, schedule='guided'):
                tid = threadid()
                is_sym = job_i // (n_X * n_Y)
                i = (job_i % (n_X * n_Y)) // n_Y
                j = job_i % n_Y

                if tid == 0:
                    with gil:
                        PyErr_CheckSignals()  # allow ^C to interrupt
                    if do_progress:
                        handle_pbar(pbar, jobs_since_last_tick, num_done)

                i_start = X_boundaries[i]
                i_end = X_boundaries[i + 1]
                num_i = i_end - i_start

                if to_self and i == j:
                    if do_linear:
                        _linear(linear_Bs, dim, num_i,
                                X_rhos_stacked[i_start:i_end], clamp,
                                outputs[linear_pos, :, i, j, 0])

                    if do_kl:
                        outputs[kl_pos, :, i, j, 0] = 0

                    if do_alpha:
                        for k in range(alpha_pos.shape[0]):
                            outputs[alpha_pos[k], :, i, j, 0] = 1

                    # no need to set js self-values to nan, they already are
                else:
                    j_start = Y_boundaries[j]
                    j_end = Y_boundaries[j + 1]
                    num_j = j_end - j_start

                    if is_sym:
                        num_p = num_j
                        p_feats = &Y_feats[j_start, 0]

                        num_q = num_i
                        q_idx = index_array[i]
                    else:
                        num_p = num_i
                        p_feats = &X_feats[i_start, 0]

                        num_q = num_j
                        q_idx = index_array[n_X + j]


                    # do the nearest neighbor search from p to q
                    flann_find_nearest_neighbors_index_float(
                        index_id=q_idx,
                        testset=p_feats,
                        trows=num_p,
                        indices=&idx_out[tid, 0, 0],
                        dists=&dists_out[tid, 0, 0],
                        nn=max_K,
                        flann_params=&params)
                    for a in range(num_p):
                        for k in range(n_Ks):
                            neighbors[tid, a, k] = fmax(
                                min_dist, sqrt(dists_out[tid, a, Ks[k] - 1]))

                    if do_linear:
                        _linear(linear_Bs, dim, num_q,
                                neighbors[tid, :num_p, :], clamp,
                                outputs[linear_pos, :, i, j, is_sym])

                    if do_kl:
                        kl(dim, num_q,
                           Y_rhos_stacked[j_start:j_end] if is_sym
                               else X_rhos_stacked[i_start:i_end],
                           neighbors[tid, :num_p, :],
                           clamp,
                           outputs[kl_pos, :, i, j, is_sym])

                    if do_alpha:
                        _alpha_div(alpha_omas, alpha_Bs, dim, num_q,
                                   Y_rhos_stacked[j_start:j_end] if is_sym
                                       else X_rhos_stacked[i_start:i_end],
                                   neighbors[tid, :num_p, :],
                                   clamp,
                                   alpha_pos, outputs[:, :, i, j, is_sym])

                    if do_js:
                        _jensen_shannon_core(
                            Ks, dim, js_min_i, js_digamma_vals, num_q,
                            all_Y_rhos_stacked[j_start:j_end] if is_sym
                                else all_X_rhos_stacked[i_start:i_end],
                            dists_out[tid, :num_p, :],
                            js_Ks_order, min_sq_dist,
                            alphas_tmp[tid],
                            outputs[js_pos, :, i, j, is_sym])
                if do_progress:
                    num_done[tid] += 1

    finally:
        free(index_array)

    out = np.asarray(outputs)

    if to_self and do_any_sym:
        w = np.asarray(do_sym_a, dtype=bool)
        out[w, ..., 1] = out[w, ..., 0].transpose(0, 1, 3, 2)

    if do_progress:
        pbar.finish()
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint handle_pbar(object pbar, long * jobs_since_last_tick,
                      long[::1] num_done) nogil except 1:
    jobs_since_last_tick[0] += 1
    cdef long done_count = 0

    if jobs_since_last_tick[0] >= 20:  # TODO: tweak? do it based on time?
        for k in range(num_done.shape[0]):
            done_count += num_done[k]

        with gil:
            pbar.update(done_count)
        jobs_since_last_tick[0] = 0
