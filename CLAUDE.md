# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hallucination detection system for AI-generated responses. The system analyzes AI responses by breaking them into claims, verifying those claims against external sources, and managing the verification workflow through a ticket system.

## Database Architecture

The system uses PostgreSQL with the following core workflow:

1. **Request Flow**: `users` → `requests` → `responses` (via `models`)
2. **Verification Flow**: `responses` → `claims` → `proofs` → `tickets`
3. **Monitoring**: `metrics` table for telemetry

### Key Tables

- `users`: System users with roles and contact info (jsonb)
- `requests`: User queries with session metadata and source tracking
- `models`: LLM configurations (name, version, params jsonb)
- `responses`: Generated answers with checksums and telemetry
- `claims`: Individual factual statements extracted from responses with scoring:
  - `raw_score`: Direct model output
  - `calibrated_score`: Adjusted confidence
  - `check_worthiness`: Priority for verification
- `proofs`: Evidence from external sources with retrieval timestamps
- `tickets`: Moderation workflow for disputed/uncertain claims
- `metrics`: Time-series event logging

### Database Schema Location

Database models are defined in `models.psql` (previously named `tables.psql`). This file is not in the current working directory but exists in git history.

## Development Environment

The project is Python-based, as indicated by the Python-specific .gitignore. Expected development tools:
- Jupyter notebooks for experimentation and analysis
- PostgreSQL with UUID extension (`uuid_generate_v4()`)
- Likely uses ML/NLP libraries for claim extraction and scoring

## Repository Context

This is a university project for "Технологии проектирования и сопровождения информационных систем" (Information Systems Design and Maintenance Technologies).
