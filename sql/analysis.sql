
-- 1. Average rentals by hour
SELECT
    hr,
    ROUND(AVG(cnt), 2) AS average_rentals
FROM bike_rentals
GROUP BY hr
ORDER BY hr;


-- 2. Average rentals by day of week
SELECT
    weekday,
    ROUND(AVG(cnt), 2) AS average_rentals
FROM bike_rentals
GROUP BY weekday
ORDER BY weekday;


-- 3. Average rentals by month
SELECT
    mnth,
    ROUND(AVG(cnt), 2) AS average_rentals
FROM bike_rentals
GROUP BY mnth
ORDER BY mnth;


-- 4. Working vs non-working days
SELECT
    workingday,
    hr,
    ROUND(AVG(cnt), 2) AS average_rentals
FROM bike_rentals
GROUP BY workingday, hr
ORDER BY workingday, hr;
