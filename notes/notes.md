# Job scheduling

The following are simple examples of metric we need to define in a job scheduling script. The first one is taking one node entirely:

```shell
#!/bin/bash
#SBATCH --time=0-60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL
```

The second one is less resource consuming:

```shell
#!/bin/bash
#SBATCH --time=0-60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2048M
#SBATCH --mem=64G
#SBATCH --account=def-yuanzhu
#SBATCH --mail-user=af4166@mun.ca
#SBATCH --mail-type=ALL
```

# PhoneLab

## Statistics

We identified 277 devices (i.e. users) from scan logs and 274 in connect logs, therefore, we created device names or user names from the scan folder.



## Database

After creating DB, first thing was to clean noise related to dates. We know the experiment was between 2014-11-07 and 2015-04-03, thus we remove any data instance out of this range by:

```sqlite
SELECT * FROM logs WHERE date > '2015-04-03' OR date < '2014-11-07' ORDER BY date;
```

We found 44789 rows, and deleted them by:

```sqlite
DELETE FROM logs WHERE date > '2015-04-03' OR date < '2014-11-07';
```
