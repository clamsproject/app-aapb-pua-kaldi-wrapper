# Evaluation plan 


## Overview 
We will evaluate the ASR performance based on [word error rate](https://en.m.wikipedia.org/wiki/Word_error_rate). For reference transcript that automatic transcription is evaluated against, we will use 17 manually corrected transcripts from across different AAPB collections. 

## Evaluation software
We will use NIST's `sclite` software to automatically compute the error rate. Evaluation script will be dockerized as an image where `sclite` is precompiled. 


