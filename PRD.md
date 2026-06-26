# PRODUCT REQUIREMENTS DOCUMENT

## AI-Powered Digital Twin of India’s Climate

**முகில்நோக்கு**

Prepared for
PS-5 — Bharatiya Antariksh Hackathon 2026

Submitted by
<NAME OF THE CANDIDATE(S) – REG. NO.>

under the technical guidance of
Dr. Kandula V Subrahmanyam • Mr. Syed Shadab • Mr. C. Sarat
(NRSC / ISRO Mentor Panel)

### 1. Executive Summary
முகில்நோக்கு is a proof-of-concept AI-powered digital twin for climate analysis over the Western Ghats. It fuses ground-based IMD gridded rainfall and temperature data with ISRO MOSDAC INSAT-3DR satellite products to generate one-step-ahead forecasts of rainfall and thermal conditions.

### 2. Product Name & Identity
Final product name: **முகில்நோக்கு**
The name uses Tamil vocabulary: “முகில்” means cloud and “நோக்கு” means view, observation, or directed attention. Together, the name communicates a climate-observation system focused on clouds, rainfall, and weather intelligence.

### 3. Problem Statement & Background
PS-5 of the Bharatiya Antariksh Hackathon 2026 asks teams to build an AI-powered digital twin of India’s climate. The expected workflow covers problem definition, data ingestion, data preprocessing, model development, visualization, and what-if simulations using satellite and ground-based datasets.

### 4. Goals & Objectives
Primary goal: deliver a working proof of concept that ingests fused IMD + INSAT data for the Western Ghats, trains a ConvLSTM forecasting model, validates predictions against historical observations, and exposes the forecast through an interactive 3D what-if dashboard.

### 5. Scope
**In Scope:**
• Pilot region: Western Ghats bounding box only.
• Variables: rainfall, maximum/minimum temperature, land-surface temperature, sea-surface temperature.
• Model: lightweight single-layer ConvLSTM.

### 7. System Architecture Overview
The architecture deliberately separates heavy computation from local interaction. Google Colab performs ingestion, preprocessing, model training, and inference. Google Drive provides persistent storage. The local workstation opens the dashboard through a secure tunnel.
