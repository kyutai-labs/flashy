# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0a]

Improve distributed with coalesced calls, and removing  synchronization points.
Add new TensorAverager that also eliminates sync points when averaging metrics.
Add support for using a TensorAverager in the LogProgressBar to also avoid sync points
unless when actually logging.

## [0.0.3]

Fix wandb audio logging

## [0.0.2] - 2023-05-24

Forgot to add MANIFEST.in so 0.01 was broken...

## [0.0.1] - 2023-05-23

Initial release.

Added broadcast for any object type.
