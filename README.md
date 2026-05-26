

# Project Almanac-FRYE: Enterprise Continuous Pricing Engine ✈️

*Flight Revenue Yield Engine - The "Amadeus & other Legacy GDS  Killer" — Moving airlines from static legacy fare buckets to AI-driven Enterprise grade continuous dynamic pricing.*

---

## 1. Introduction, Pricing Models of Amadeus, & The Problem Statement

### Introduction
For decades, the airline industry has relied on Global Distribution Systems (GDS) and Passenger Service Systems (PSS) legacy core providers like Amadeus, Sabre, and Travelport.
**Project Almanac-FRYE** is an enterprise-grade Machine Learning pipeline (built on PyTorch and Hugging Face Accelerate) designed to decouple an airline's pricing intelligence from legacy PSS/GDS software, bringing real-time continuous pricing in-house.

### The Legacy Pricing Model (Amadeus Altéa RM, Sabre AirVision, Travelport RM)
Traditionally, legacy PSS and GDS suites price inventory using **Static Fare Buckets**. A flight is divided into 26 alphabetical classes (e.g., J for Business, Y for Full Economy, B for Basic). Pricing logic operates on rigid, rule-based thresholds managed by core infrastructure providers:
* *Rule A:* If lead time < 21 days, open the $400 'M' bucket.
* *Rule B:* If lead time > 21 days, open the $250 'Q' bucket.

### The Problem Statement
This legacy model creates three massive financial leaks for airlines:
1.  **Margin Leakage (Leaving Money on the Table):** If a customer is willing to pay $450, but the nearest open bucket dictated by the core provider is $400, the airline loses $50 of pure margin.
2.  **Lost Bookings:** If a competitor prices their flight at $390, and the airline's bucket is rigidly stuck at $400 due to legacy system limitations, the airline loses the entire sale.
3.  **The Late Bucket Reset Latency:** Legacy RM systems (such as those from Amadeus, Sabre, or Travelport) typically run batch updates overnight or at fixed multi-hour intervals. If a market shift occurs at 9:00 AM (e.g., a competitor drops a price or a sudden surge in demand happens), these systems keep the airline locked in old price buckets for hours. This delay results in either hours of uncompetitive pricing or severe underpricing before the system finally triggers a reset.
---

## 2. What the Solution is

**Almanac-FRYE** is a Continuous Dynamic Pricing Engine powered by a **Time-Series Transformer Neural Network**.

Instead of relying on legacy providers like Amadeus, Sabre, or Travelport for both pricing and distribution, the airline uses these systems *strictly as a dumb pipe* for distribution. The Almanac-FRYE engine runs internally on AWS SageMaker, bypassing the 26-letter bucket system entirely to calculate the exact, continuous mathematical optimum price for every single transaction (e.g., $412.36 instead of $400.00).

**Instantaneous Micro-Pricing Elimination of Latency:** Almanac-FRYE operates in continuous time, generating real-time price updates on-the-fly for every individual search request. By eliminating the multi-hour or overnight batch processing wait times characteristic of legacy systems, the airline avoids stale inventory positioning and captures transient windows of maximum consumer willingness-to-pay instantly.
---

## 3. How It Works

The architecture transitions from simple batch-processing to a distributed Machine Learning Operations (MLOps) pipeline:

* **The Data Lake (14 Enterprise Parameters):** The model ingests live and historical data across four primary dimensions:
    * *Timing & Demand* (Booking Window, Day of Week, Seasonality, Time of Flight)
    * *Route & Competition* (Popularity, Direct vs. Connecting, Carrier Competition, Airport Hub)
    * *Fare Class & Inventory* (Bucket Tier, Service Class, Ancillaries)
    * *Macro-Economic & Ops* (Fuel Prices, Taxes, Currency Exchange)
* **The Normalization Wall:** Real-world metrics are standardized using `scikit-learn`'s `MinMaxScaler` before hitting the neural network, allowing the model to quickly converge on a highly accurate 0.0-1.0 scale.
* **The "Brain" (Time-Series Transformer):** The engine processes a rolling window of the previous 99 searches to predict the optimal price for the 100th search in real-time, utilizing Hugging Face's `TimeSeriesTransformerModel`.
* **Route-Sharded Distributed Training:** Deployed on AWS SageMaker `ml.g4dn.xlarge` instances, individual models are trained per route (e.g., JFK-LHR vs. BLR-DEL) to capture specific price elasticities.
* **Live Inference:** A Multi-Model Endpoint (MME) un-squashes the AI's prediction back into nominal currency and pushes the final absolute price to the OTA via the distribution networks in milliseconds.

---

## 4. The Financial Architecture: Shifting from High OpEx to CapEx/OpEx Efficiency

Legacy pricing solutions force airlines into a highly expensive, vendor-locked financial model. PSS and GDS providers (including Amadeus, Sabre, and Travelport) typically charge a "Per Passenger Boarded" (PB) or transaction-based fee for their Revenue Management modules. This represents a **100% High-Cost Operational Expenditure (OpEx)** that scales linearly with passenger volume—meaning the more successful the airline is, the more expensive the software becomes.

**Project Almanac-FRYE fundamentally restructures this IT balance sheet:**
* **Strategic CapEx (Intellectual Property Generation):** The initial build requires Capital Expenditure (CapEx) investment in specialized MLOps talent, data engineering, and foundational model training. Instead of renting a generic algorithm from a legacy provider, the airline is building a proprietary, appreciating corporate asset.
* **Optimized, Decoupled OpEx:** The ongoing operational costs shift to AWS Cloud infrastructure (SageMaker inference endpoints, S3 storage) and third-party data API feeds. Crucially, this cloud OpEx scales with *compute and inference efficiency*, not passenger volume.

By decoupling the cost of the pricing engine from the number of tickets sold, airlines achieve massive economies of scale. The cost per transaction approaches zero over time, transforming the pricing system from a heavy, linear vendor toll into a highly efficient, proprietary profit center.

---

## 5. The Value Provided (Strategic Pitches)

When pitching this infrastructure to the C-Suite, it is not framed as an IT cost-saving measure, but as a multi-billion dollar financial lever:

### I. The "Margin Multiplier" (The CFO Pitch)
Because the operational costs of flying (fuel, crew, aircraft) are already sunk, the estimated 3% top-line revenue uplift generated by Almanac-FRYE is practically 100% gross margin. This means a 3% revenue bump can double or triple an airline's actual Net Profit.

### II. Breaking the "Vendor Parity" Trap (The CEO Pitch)
As long as an airline outsources its pricing to the same legacy software vendor its competitors use, it brings a knife to a knife fight. Almanac-FRYE allows the airline to operate in "Continuous Time." When competitors make overnight batch adjustments on platforms like Sabre, Amadeus, or Travelport, Almanac-FRYE reacts in real-time, weaponizing pricing to steal market share while rivals are asleep.

### III. The "Unbundling" Engine (The CMO Pitch)
Legacy GDS buckets force airlines to sell basic seats. Almanac-FRYE is an offer-creation engine. When the AI detects a corporate traveler booking 48 hours out, it doesn't just raise the base fare—it automatically constructs a bespoke, high-margin 'Premium Corporate Bundle' (flexibility + Wi-Fi), shifting the airline into true digital retailing and breaking past old legacy system distribution limits.

---

## 6. Gain and Impact: The Top 25 Global Airlines

Applying a conservative **3% Gross Revenue Uplift** (minus a heavy **$5M Annual AI MLOps Cost**), here is the net financial impact Almanac-FRYE would have on the world's leading airlines.

*Notice how for low-margin legacy or domestic carriers, this engine completely alters corporate valuation by doubling or tripling absolute net income.*

| Global Rank | Airline / Aviation Group | Est. Annual Revenue | Est. Baseline Net Profit | Net Profit Impact (Less $5M AI Cost) | Net Profit % Increase | Most Effective Strategic Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Delta Air Lines** | $63.0 Billion | $5.00 Billion | +$1.88 Billion | **+37.6%** | **The Margin Multiplier:** Expands industry-leading margins to record heights. |
| **2** | **United Airlines** | $59.0 Billion | $3.35 Billion | +$1.76 Billion | **+52.5%** | **Unbundling Engine:** Maximizes premium cabins via high-margin corporate bundles. |
| **3** | **American Airlines** | $54.5 Billion | $0.80 Billion | +$1.62 Billion | **+202.5%** | **Breaking Vendor Parity:** Triples company profitability by correcting thin margins. |
| **4** | **Lufthansa Group** | $45.0 Billion | $1.50 Billion | +$1.34 Billion | **+89.3%** | **The Margin Multiplier:** Directly counteracts soaring regional labor and airport taxes. |
| **5** | **IAG (British Airways)** | $38.0 Billion | $1.20 Billion | +$1.13 Billion | **+94.2%** | **Unbundling Engine:** Optimizes long-haul premium segments vs. low-cost operators. |
| **6** | **Air France-KLM** | $36.0 Billion | $0.80 Billion | +$1.07 Billion | **+133.8%** | **Breaking Vendor Parity:** Delivers lightning-fast pricing to steal continental market share. |
| **7** | **Emirates** | $35.0 Billion | $5.60 Billion | +$1.04 Billion | **+18.6%** | **Unbundling Engine:** Captures massive long-haul superconnector route willingness-to-pay. |
| **8** | **Southwest Airlines** | $28.0 Billion | $0.60 Billion | +$835 Million | **+139.2%** | **Breaking Vendor Parity:** Protects domestic dominance against ultra-low-cost alternatives. |
| **9** | **China Southern** | $24.5 Billion | $0.40 Billion | +$730 Million | **+182.5%** | **The Margin Multiplier:** Converts massive, high-volume domestic capacity into immense profit. |
| **10** | **Air China** | $23.5 Billion | $0.40 Billion | +$700 Million | **+175.0%** | **The Margin Multiplier:** Drives hyper-efficient state-backed operational profitability. |
| **11** | **Qatar Airways** | $22.2 Billion | $1.50 Billion | +$661 Million | **+44.1%** | **Unbundling Engine:** Maximizes high-end multi-segment luxury routing and bespoke stopover yield. |
| **12** | **Turkish Airlines** | $22.1 Billion | $1.40 Billion | +$658 Million | **+47.0%** | **Breaking Vendor Parity:** Dynamically maps real-time connecting entries across a massive footprint. |
| **13** | **China Eastern** | $18.5 Billion | $0.20 Billion | +$550 Million | **+275.0%** | **The Margin Multiplier:** Drastically restructures thin margins on hyper-competitive trunk lines. |
| **14** | **Air Canada** | $16.5 Billion | $0.50 Billion | +$490 Million | **+98.0%** | **Unbundling Engine:** Captures transborder high-yield business travel via context-aware ancillaries. |
| **15** | **Ryanair Group** | $15.5 Billion | $2.20 Billion | +$460 Million | **+20.9%** | **Breaking Vendor Parity:** Weaponizes high-frequency continuous micro-adjustments to crush regional rival sales. |
| **16** | **Qantas Airways** | $14.5 Billion | $1.10 Billion | +$430 Million | **+39.1%** | **The Margin Multiplier:** Capitalizes on highly isolated international entry-points. |
| **17** | **Singapore Airlines** | $14.5 Billion | $1.80 Billion | +$430 Million | **+23.9%** | **Unbundling Engine:** Locks in corporate and luxury market segments through dynamic elite bundling tiers. |
| **18** | **All Nippon Airways** | $14.3 Billion | $0.90 Billion | +$424 Million | **+47.1%** | **The Margin Multiplier:** Protects critical regional corporate business streams from structural micro-shifts. |
| **19** | **LATAM Airlines** | $13.0 Billion | $0.80 Billion | +$385 Million | **+48.1%** | **Breaking Vendor Parity:** Secures major cross-border networks from aggressive pan-regional discount entries. |
| **20** | **Korean Air** | $12.0 Billion | $0.70 Billion | +$355 Million | **+50.7%** | **Unbundling Engine:** Accelerates post-merger capacity integration by optimizing high-demand transpacific corridors. |
| **21** | **easyJet** | $12.0 Billion | $0.40 Billion | +$355 Million | **+88.8%** | **Breaking Vendor Parity:** Drives automated price matching at high-density, slot-constrained European airports. |
| **22** | **Cathay Pacific** | $11.5 Billion | $0.60 Billion | +$340 Million | **+56.7%** | **Unbundling Engine:** Maximizes premium seat yields as network capacity limits stabilize. |
| **23** | **Japan Airlines** | $11.5 Billion | $0.60 Billion | +$340 Million | **+56.7%** | **The Margin Multiplier:** Stabilizes premium yield reliability against domestic corporate travel fluctuations. |
| **24** | **Alaska Airlines** | $11.0 Billion | $0.30 Billion | +$325 Million | **+108.3%** | **Breaking Vendor Parity:** Defends highly profitable regional business paths via instant intra-hour matching. |
| **25** | **JetBlue Airways** | $9.5 Billion | $0.10 Billion | +$280 Million | **+280.0%** | **The Margin Multiplier:** Instantly turns zero-margin budget leakage into robust, enterprise net profit. |
