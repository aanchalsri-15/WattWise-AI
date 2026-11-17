# Project methodology and viva notes

**Student:** Aanchal Srivastava  |  **Roll number:** 2025031101  |  **Institution:** Madan Mohan Malaviya University of Technology

## Objective
Predict next-day daily mean household active power in kW.

## Why regression?
The output is a continuous numeric quantity (kW), so this is regression rather than classification.

## Why Random Forest?
It can model nonlinear relationships among lagged consumption and calendar features, requires limited scaling, and provides feature importance. It is a model choice, not a guarantee of best performance.

## Why a persistence baseline?
A simple forecast using the latest observed day is a meaningful benchmark. A complex model should be compared against it.

## Why chronological split?
Random shuffling can leak future patterns into training. Earlier dates train the model and later dates test it.

## Metrics
- MAE: average absolute prediction error in kW.
- RMSE: penalizes larger errors more strongly; unit is kW.
- R²: proportion of target variance explained relative to the test-set mean baseline; can be negative.

## Limitations
- Data covers one household and is historical.
- Daily interpolation may smooth abrupt changes.
- The project does not evaluate bill savings, carbon reduction, or appliance disaggregation.
- Dashboard needs user testing and monitoring.

## Viva questions
1. Why is this a regression problem?
2. What is the target variable and its unit?
3. Why aggregate minute readings to daily values?
4. How are '?' values handled?
5. What is feature engineering?
6. How is data leakage avoided?
7. Why use a chronological split?
8. What does MAE mean?
9. Why compare with a persistence baseline?
10. What are the dataset limitations?
