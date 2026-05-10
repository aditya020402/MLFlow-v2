## Task Type
clustering

## Target Variable
None

## Business Context
Segment ships into operational performance groups based on voyage metrics
such as speed, engine power, fuel efficiency, cargo load, cost, and revenue.
Use clusters to identify high-efficiency vs high-cost ship profiles for
route optimization and maintenance planning.

## Validation
yes

## ML Requirements
Try KMeans and AgglomerativeClustering. Prefer 3-6 clusters.
Exclude Date column. Encode categorical columns (Ship_Type, Route_Type,
Engine_Type, Maintenance_Status, Weather_Condition) before clustering.

## Evaluation Criteria

