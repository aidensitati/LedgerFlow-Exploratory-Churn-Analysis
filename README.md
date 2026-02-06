# The Sensitivity of High-Tier Clients in SaaS Products

Explore how intake configurations shape churn risk in SaaS, with a focus on enterprise clients and high pricing exposure.
  
This repository consolidates the LedgerLoop exploratory churn analysis, highlighting structural intake risks and cohort dynamics.


## Project Stages

### 1. Dataset Consolidation
**Goal:** Combine customer, subscription, and revenue data into a canonical table.  
**Outcome:** Single customer-level dataset with tenure and coverage flags; ready for analysis.

### 2. Data Integrity and Leakage Risk
**Goal:** Validate data integrity and remove post-outcome contamination.  
**Outcome:** Leakage-neutral dataset suitable for exploratory analysis.

### 3. Target Variable Analysis
**Goal:** Understand the churn_flag and lifecycle dynamics.  
**Outcome:** Churn administrative, right-censored; analysis reframed to lifecycle timing.

### 4. Feature Triage
**Goal:** Identify a minimal, high-integrity feature set.  
**Outcome:** Structural features retained; predictive surfaces avoided.

### 5. Diagnostic Exploration
**Goal:** Identify intake bottlenecks and cohort anomalies.  
**Outcome:** February flagged as anomalous; structural intake conditions align with later churn.

### 6. Inferential Exploration
**Goal:** Assess pre-churn intake conditions for risk plausibility.  
**Outcome:** February disproportionately onboarded enterprise/high-price clients, revealing systemic segmentation and pricing vulnerability.

### 7. Key Insight
Churn at LedgerLoop is largely shaped at intake, with elevated risk concentrated in enterprise customers exposed to higher pricing structures; February surfaced this risk because it disproportionately onboarded such configurations.
