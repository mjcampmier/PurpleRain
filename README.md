# PurpleRain
An open source PurpleAir data scraper, manager and visualizer.

Welcome to PurpleRain. PurpleRain allows users to automate data retrivals from PurpleAir's database at http://www.purpleair.com/sensorlist
The python script PurpleRain.py contains most of the functions within the module, except for jdutil.py, a script for manipulating Julian
Date data structures. Please review at PurpleAir Tutorial.ipynb or the commented documentation in the .py scripts before submitting
issues. Also view the .pngs to preview the visualizations. Feel free to use the sample csv files to test out the calibration import
and large sensor download automation options.

jdutil.py is a slightly modified version of the open source, free to share code by @jiffyclub - http://github.com/jiffyclub



## Dependencies
PurpleRain requires the following python3 libraries (anaconda should cover all):
numpy, pandas, scipy, time, datetime, math, h5py, selenium, matplotlib, calendar, sklearn & dateutil

PurpleRain requires the following files in addition to the repo:
Google Chrome: https://www.google.com/chrome/?brand=CHBD&gclid=CjwKCAjwnf7qBRAtEiwAseBO_NoaYQIW1syc1yLs7mmi5OGaKOIYO0ZvpGZ-8BPEZfPEAG-dDy49WRoC9LgQAvD_BwE&gclsrc=aw.ds

Google Chrome Driver: https://chromedriver.chromium.org/

Make sure your versions of Google Chrome and Google Chrome Driver are synchronized.

## Suggested Resources
#### PurpleAir - manufacuturer of the PurpleAir sensor: https://www2.purpleair.com/
#### Panoply - NASA app for viewing HDF/H5 files: https://www.giss.nasa.gov/tools/panoply/download/
#### Iowa State Mesonet - Excellent source of weather data: http://www.mesonet.agron.iastate.edu/request/download.phtml
#### Wikipedia Article on Timezones - contains all Timezone codes: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
#### Apte Group UT Austin - the research group I am apart of: http://apte.caee.utexas.edu/
