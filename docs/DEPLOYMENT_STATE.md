# ATLAS Deployment State

This file is the persistent source of truth for development changes waiting to be deployed to the ATLAS Jetson.

## Jetson

- **Status:** UNKNOWN
- **Last user-confirmed state:** Not recorded
- **Last updated:** 2026-08-30

## Last Successful Deployment

- **Commit/version:** Not recorded
- **Date:** Not recorded
- **Verified:** Not recorded

## Pending Deployment

None recorded.

## Files Added / Removed / Modified

None recorded.

## Configuration / Environment Changes

None recorded.

## Dependency Changes

None recorded.

## Services / Containers

None recorded.

## Models / Assets / Migrations

None recorded.

## Offline Tests Completed

None recorded.

## Jetson-Only Tests Pending

None recorded.

## Known Deployment Risks

None recorded.

## Status Definitions

- **Implemented** — code/configuration changed.
- **Committed** — changes recorded in Git.
- **Pushed** — commit exists on the remote repository.
- **Deployed** — the target Jetson received the intended version.
- **Verified** — the deployed version was tested successfully.

## Operating Rule

**OFFLINE = develop, test locally where possible, and accumulate pending deployment state.**

**ONLINE = deploy all approved pending changes, then verify them.**

Do not deploy after the user has stated that the Jetson is offline unless the user later explicitly states that it is available again.
