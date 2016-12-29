from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context


inter_subject_eval = CV(crossval_type='inter-subject', batch_size=1000)
inter_session_eval = CV(crossval_type='inter-session', batch_size=1000)
one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)

print('Inter-session CSL-HDEMG')
print('============')

with Context(parallel=True, level='DEBUG'):
    acc = inter_session_eval.accuracies(
        [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             Mod=dict(num_gesture=27,
                      adabn=True,
                      num_adabn_epoch=10,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      params='.cache/sensors-csl-inter-session-%d/model-0028.params'))],
        folds=np.arange(25))
    print('Per-trial majority voting accuracy: %f' % acc.mean())

print('')
print('Inter-subject CapgMyo DB-b')
print('============')

with Context(parallel=True, level='DEBUG'):
    acc = inter_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('dbb'),
             Mod=dict(num_gesture=8,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      params='.cache/sensors-dbb-inter-subject-%d/model-0028.params'))],
        folds=np.arange(10),
        windows=[1, 150])
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('150 frames (150 ms) majority voting accuracy: %f' % acc[1])

print('')
print('Inter-session CapgMyo DB-b')
print('============')

with Context(parallel=True, level='DEBUG'):
    acc = inter_session_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('dbb'),
             Mod=dict(num_gesture=8,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      params='.cache/sensors-dbb-inter-session-%d/model-0028.params'))],
        folds=np.arange(10),
        windows=[1, 150])
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('150 frames (150 ms) majority voting accuracy: %f' % acc[1])

print('')
print('Inter-subject CapgMyo DB-c')
print('============')

with Context(parallel=True, level='DEBUG'):
    acc = inter_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('dbc'),
             Mod=dict(num_gesture=12,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      params='.cache/sensors-dbc-inter-subject-%d/model-0028.params'))],
        folds=np.arange(10),
        windows=[1, 150])
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('150 frames (150 ms) majority voting accuracy: %f' % acc[1])

print('')
print('Inter-subject NinaPro DB1')
print('===========')
with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('ninapro-db1/caputo'),
             Mod=dict(num_gesture=52,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                      params='.cache/sensors-ninapro-one-fold-intra-subject-%d/model-0028.params'))],
        folds=np.arange(27),
        windows=[1, 40])
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('40 frames (400 ms) majority voting accuracy: %f' % acc[1])
