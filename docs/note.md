# Note from analysis
## Database
After creating DB, first thing was to clean noise related to dates. We know the experiment was between 2014-11-07 and 2015-04-03, thus we remove any data instance out of this range by:
```sqlite
SELECT * FROM logs WHERE date > '2015-04-03' OR date < '2014-11-07' ORDER BY date;
```
We found 44789 rows, and deleted them by:
```sqlite
DELETE FROM logs WHERE date > '2015-04-03' OR date < '2014-11-07';
```

