# Custom Email Alerts

This section explains who to add a custom alert four your institution.

Copy the most similar existing alert:

```
--custom-1
--custom-2
--custom-3
```

CustomAlert(Base)

```
$ cd job_defense_shield/alert
$ cp <most-similar-alert>.py CustomAlert1.py
```

Use a text editor to write the alert.

Create an new entry in the configuration file:

```
custom-alert-1:
  clusters:
    - della
  partitions:
    - cpu
    - gpu
    - bigmem
```

Then run with:

```
$ job_defense_shield --custom-1 --days=10 --email
```

You can add any attribute to config.yaml and it will be available in the alert.
