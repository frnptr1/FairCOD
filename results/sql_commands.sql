-- Retrieve cases where the wmae is minimum
-- For OLA Backlog
SELECT id, DPC, aggregation, Weighting, tot_error, WMAE, N_WMAE
FROM results
WHERE Backlog = 'OLA'
ORDER BY N_WMAE ASC
LIMIT 5;

-- For Websites Backlog
SELECT id, DPC, aggregation, Weighting, tot_error, WMAE, N_WMAE
FROM results
WHERE Backlog = 'Websites'
ORDER BY N_WMAE ASC
LIMIT 5;

-- For Online Sales Backlog
SELECT id, DPC, aggregation, Weighting, tot_error, WMAE, N_WMAE
FROM results
WHERE Backlog = 'Online Sales'
ORDER BY N_WMAE ASC
LIMIT 5;


-- Get detailed info of a specific campaign with minimum error
SELECT r.aggregation, r.DPC, r.Backlog, r.tot_error, r.WMAE, r.N_WMAE,
       c.epic_name, c.epic_id, c.true_cod, c.pred_cod, c.class_difference, c.error
FROM results r
JOIN cod_estimation c ON r.id = c.results_id
WHERE r.id = "20240716_173347_685800_23.4_Online Sales"
ORDER BY c.error DESC;


-- Get detailed info about specific backlog and dpc
SELECT r.aggregation, r.DPC, r.Backlog, r.tot_error, r.WMAE, r.N_WMAE,
       c.epic_name, c.epic_id, c.true_cod, c.pred_cod, c.class_difference, c.error
FROM results r
JOIN cod_estimation c ON r.id = c.results_id
WHERE r.backlog = "OLA" AND r.dpc = 24.3
ORDER BY c.error DESC;