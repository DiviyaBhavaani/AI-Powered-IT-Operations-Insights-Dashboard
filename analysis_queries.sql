-- 1) Monthly incident volume
SELECT month, COUNT(*) AS incidents
FROM incidents
GROUP BY month
ORDER BY month;

-- 2) Closed-incident breach rate by category
SELECT
    category,
    ROUND(AVG(CASE WHEN status = 'Closed' THEN breached_sla END), 4) AS breach_rate
FROM incidents
GROUP BY category
ORDER BY breach_rate DESC;

-- 3) Team backlog snapshot
SELECT
    assigned_team,
    COUNT(*) AS open_incidents
FROM incidents
WHERE status = 'Open'
GROUP BY assigned_team
ORDER BY open_incidents DESC;

-- 4) Services with highest closed-incident average resolution time
SELECT
    service,
    ROUND(AVG(resolution_hours), 2) AS avg_resolution_hours
FROM incidents
WHERE status = 'Closed'
GROUP BY service
ORDER BY avg_resolution_hours DESC
LIMIT 10;

-- 5) Severity mix
SELECT
    severity,
    COUNT(*) AS incidents
FROM incidents
GROUP BY severity
ORDER BY severity;
