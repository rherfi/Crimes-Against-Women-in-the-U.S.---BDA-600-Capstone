ALTER TABLE nibrs_victim ALTER COLUMN age_num TYPE varchar;
ALTER TABLE nibrs_weapon ALTER COLUMN weapon_id TYPE integer;

ALTER TABLE nibrs_arrestee_weapon DISABLE TRIGGER ALL;
ALTER TABLE nibrs_bias_motivation DISABLE TRIGGER ALL;
ALTER TABLE nibrs_incident DISABLE TRIGGER ALL;
ALTER TABLE nibrs_offense DISABLE TRIGGER ALL;
ALTER TABLE nibrs_victim DISABLE TRIGGER ALL;
ALTER TABLE nibrs_victim_circumstances DISABLE TRIGGER ALL;
ALTER TABLE nibrs_victim_injury DISABLE TRIGGER ALL;
ALTER TABLE nibrs_victim_offender_rel DISABLE TRIGGER ALL;
ALTER TABLE nibrs_weapon DISABLE TRIGGER ALL;


COPY NIBRS_ACTIVITY_TYPE FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_ACTIVITY_TYPE.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_AGE FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_AGE.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_LIST FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_BIAS_LIST.csv' DELIMITER ',' HEADER CSV encoding 'windows-1251';
COPY NIBRS_LOCATION_TYPE FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_LOCATION_TYPE.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_OFFENSE_TYPE FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_OFFENSE_TYPE.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_CIRCUMSTANCES FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_ETHNICITY FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_ETHNICITY.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_INJURY FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_RELATIONSHIP FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_RELATIONSHIP.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_WEAPON_TYPE FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/NIBRS_WEAPON_TYPE.csv' DELIMITER ',' HEADER CSV;
COPY REF_RACE FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/1. MASTER DEFINITION FILES/REF_RACE.csv' DELIMITER ',' HEADER CSV;


-- Tip: wrap in a transaction if you want all-or-nothing
BEGIN;

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/AK-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/AL-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/AR-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/AZ-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/CA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/CO-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/CT-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/DC-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/DE-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/FL-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/GA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/HI-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/IA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/ID-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/IL-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/IN-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/KS-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/KY-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/LA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MD-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/ME-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MI-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MN-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MO-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MS-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/MT-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NC-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/ND-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NE-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NH-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NJ-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NM-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NV-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/NY-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/OH-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/OK-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/OR-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/PA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/RI-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/SC-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/SD-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/TN-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/TX-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/UT-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/VA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/VT-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/WA-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/WI-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/WV-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COPY public.agencies
FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/3. Agencies All States/WY-agencies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'WIN1252');

COMMIT;

COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/AK-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/AL-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/AR-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/AZ-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/CA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/CO-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/CT-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/DC-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/DE-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/FL-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/GA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/HI-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/IA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/ID-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/IL-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/IN-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/KS-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/KY-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/LA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MD-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/ME-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MI-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MN-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MO-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MS-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/MT-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NC-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/ND-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NE-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NH-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NJ-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NM-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NV-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/NY-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/OH-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/OK-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/OR-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/PA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/RI-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/SC-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/SD-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/TN-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/TX-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/UT-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/VA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/VT-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/WA-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/WI-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/WV-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;
COPY NIBRS_BIAS_MOTIVATION FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/5. Bias Motivation/WY-NIBRS_BIAS_MOTIVATION.csv' DELIMITER ',' HEADER CSV;


COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/AK-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/AL-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/AR-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/AZ-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/CA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/CO-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/CT-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/DC-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/DE-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/FL-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/GA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/HI-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/IA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/ID-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/IL-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/IN-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/KS-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/KY-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/LA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MD-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/ME-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MI-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MN-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MO-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MS-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/MT-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NC-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/ND-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NE-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NH-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NJ-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NM-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NV-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/NY-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/OH-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/OK-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/OR-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/PA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/RI-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/SC-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/SD-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/TN-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/TX-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/UT-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/VA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/VT-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/WA-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/WI-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/WV-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_incident FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/6. Incident All States/WY-NIBRS_incident.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/AK-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/AL-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/AR-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/AZ-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/CA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/CO-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/CT-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/DC-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/DE-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/FL-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/GA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/HI-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/IA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/ID-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/IL-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/IN-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/KS-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/KY-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/LA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MD-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/ME-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MI-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MN-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MO-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MS-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/MT-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NC-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/ND-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NE-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NH-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NJ-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NM-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NV-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/NY-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/OH-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/OK-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/OR-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/PA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/RI-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/SC-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/SD-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/TN-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/TX-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/UT-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/VA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/VT-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/WA-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/WI-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/WV-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_offense FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/7. Offense All States/WY-NIBRS_OFFENSE.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/AK-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/AL-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/AR-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/AZ-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/CA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/CO-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/CT-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/DC-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/DE-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/FL-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/GA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/HI-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/IA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/ID-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/IL-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/IN-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/KS-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/KY-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/LA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MD-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/ME-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MI-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MN-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MO-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MS-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/MT-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NC-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/ND-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NE-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NH-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NJ-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NM-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NV-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/NY-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/OH-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/OK-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/OR-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/PA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/RI-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/SC-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/SD-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/TN-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/TX-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/UT-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/VA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/VT-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/WA-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/WI-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/WV-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/11. Victim All States/WY-NIBRS_VICTIM.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/AK-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/AL-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/AR-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/AZ-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/CA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/CO-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/CT-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/DC-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/DE-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/FL-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/GA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/HI-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/IA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/ID-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/IL-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/IN-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/KS-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/KY-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/LA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MD-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/ME-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MI-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MN-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MO-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MS-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/MT-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NC-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/ND-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NE-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NH-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NJ-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NM-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NV-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/NY-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/OH-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/OK-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/OR-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/PA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/RI-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/SC-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/SD-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/TN-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/TX-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/UT-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/VA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/VT-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/WA-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/WI-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/WV-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_circumstances FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/8. Victim Circumstances/WY-NIBRS_VICTIM_CIRCUMSTANCES.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/AK-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/AL-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/AR-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/AZ-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/CA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/CO-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/CT-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/DC-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/DE-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/FL-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/GA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/HI-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/IA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/ID-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/IL-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/IN-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/KS-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/KY-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/LA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MD-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/ME-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MI-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MN-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MO-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MS-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/MT-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NC-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/ND-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NE-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NH-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NJ-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NM-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NV-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/NY-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/OH-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/OK-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/OR-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/PA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/RI-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/SC-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/SD-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/TN-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/TX-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/UT-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/VA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/VT-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/WA-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/WI-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/WV-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_injury FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/9. Victim Injury All States/WY-NIBRS_VICTIM_INJURY.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/AK-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/AL-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/AR-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/AZ-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/CA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/CO-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/CT-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/DC-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/DE-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/FL-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/GA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/HI-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/IA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/ID-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/IL-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/IN-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/KS-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/KY-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/LA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MD-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/ME-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MI-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MN-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MO-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MS-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/MT-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NC-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/ND-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NE-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NH-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NJ-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NM-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NV-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/NY-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/OH-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/OK-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/OR-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/PA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/RI-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/SC-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/SD-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/TN-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/TX-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/UT-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/VA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/VT-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/WA-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/WI-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/WV-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_victim_offender_rel FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/10. Victim Offender Relationship All States/WY-NIBRS_VICTIM_OFFENDER_REL.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/AK-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/AL-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/AR-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/AZ-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/CA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/CO-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/CT-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/DC-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/DE-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/FL-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/GA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/HI-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/IA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/ID-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/IL-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/IN-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/KS-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/KY-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/LA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MD-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/ME-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MI-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MN-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MO-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MS-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/MT-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NC-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/ND-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NE-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NH-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NJ-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NM-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NV-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/NY-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/OH-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/OK-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/OR-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/PA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/RI-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/SC-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/SD-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/TN-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/TX-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/UT-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/VA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/VT-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/WA-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/WI-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/WV-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;
COPY nibrs_weapon FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/12. Weapons All States/WY-NIBRS_WEAPON.csv' DELIMITER ',' HEADER CSV;

COPY nibrs_agency_locations FROM '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/nibrs_agency_locations_2021.csv' DELIMITER ',' HEADER CSV;


ALTER TABLE nibrs_arrestee_weapon ENABLE TRIGGER ALL;
ALTER TABLE nibrs_bias_motivation ENABLE TRIGGER ALL;
ALTER TABLE nibrs_incident ENABLE TRIGGER ALL;
ALTER TABLE nibrs_offense ENABLE TRIGGER ALL;
ALTER TABLE nibrs_victim ENABLE TRIGGER ALL;
ALTER TABLE nibrs_victim_circumstances ENABLE TRIGGER ALL;
ALTER TABLE nibrs_victim_injury ENABLE TRIGGER ALL;
ALTER TABLE nibrs_victim_offender_rel ENABLE TRIGGER ALL;
ALTER TABLE nibrs_weapon ENABLE TRIGGER ALL;


ALTER TABLE agencies 
DROP COLUMN yearly_agency_id,
--DROP COLUMN ori,
DROP COLUMN legacy_ori,
DROP COLUMN covered_by_legacy_ori,
DROP COLUMN direct_contributor_flag,
DROP COLUMN dormant_flag,
DROP COLUMN dormant_year,
DROP COLUMN ncic_agency_name,
DROP COLUMN pub_agency_unit,
DROP COLUMN agency_status,
DROP COLUMN state_id,
DROP COLUMN state_abbr,
DROP COLUMN state_postal_abbr,
DROP COLUMN division_code,
DROP COLUMN division_name,
DROP COLUMN region_code,
DROP COLUMN region_name,
DROP COLUMN region_desc,
DROP COLUMN agency_type_name,
DROP COLUMN submitting_agency_id,
DROP COLUMN sai,
DROP COLUMN submitting_agency_name,
DROP COLUMN suburban_area_flag,
DROP COLUMN population_group_id,
DROP COLUMN population_group_code,
DROP COLUMN population_group_desc,
DROP COLUMN parent_pop_group_code,
DROP COLUMN parent_pop_group_desc,
DROP COLUMN mip_flag,
DROP COLUMN pop_sort_order,
DROP COLUMN summary_rape_def,
DROP COLUMN pe_reported_flag,
DROP COLUMN male_total,
DROP COLUMN female_total,
DROP COLUMN nibrs_cert_date,
DROP COLUMN nibrs_start_date,
DROP COLUMN nibrs_leoka_start_date,
DROP COLUMN nibrs_ct_start_date,
DROP COLUMN nibrs_multi_bias_start_date,
DROP COLUMN nibrs_off_eth_start_date,
DROP COLUMN covered_flag,
DROP COLUMN county_name,
DROP COLUMN msa_name,
DROP COLUMN publishable_flag,
DROP COLUMN participated,
DROP COLUMN nibrs_participated; 


ALTER TABLE nibrs_incident DROP COLUMN cargo_theft_flag, DROP COLUMN submission_date, DROP COLUMN report_date_flag, 
DROP COLUMN cleared_except_id, DROP COLUMN cleared_except_date, DROP COLUMN incident_status, 
DROP COLUMN data_home, DROP COLUMN orig_format, DROP COLUMN did;
ALTER TABLE nibrs_offense DROP COLUMN data_year, DROP COLUMN attempt_complete_flag, 
DROP COLUMN num_premises_entered, DROP COLUMN method_entry_code;
ALTER TABLE nibrs_victim DROP COLUMN assignment_type_id, DROP COLUMN outside_agency_id;
-- ALTER TABLE nibrs_circumstances  DROP COLUMN data_year, 
-- DROP COLUMN justifiable_force_id;
ALTER TABLE nibrs_victim_injury DROP COLUMN data_year;
ALTER TABLE nibrs_victim_offender_rel DROP COLUMN nibrs_victim_offender_id;
ALTER TABLE nibrs_weapon DROP COLUMN data_year, 
DROP COLUMN nibrs_weapon_id;


DELETE FROM nibrs_victim
WHERE sex_code <>'F';

DELETE FROM nibrs_victim
WHERE age_id < 21;

-- SELECT * 
-- FROM nibrs_victim;

CREATE TABLE victim_injury_and_victim AS
SELECT

  -- all the other nibrs_victim_injury columns you need
  vi.injury_id,
  -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM public.nibrs_victim_injury AS vi
JOIN public.nibrs_victim AS v
  ON vi.victim_id = v.victim_id;

--SELECT * FROM victim_injury_and_victim;

CREATE TABLE victim_injury_w_id AS
SELECT

  -- all the other nibrs_victim_injury columns you need
  vi.injury_code,
  vi.injury_name,
  -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_and_victim AS v
JOIN public.nibrs_injury AS vi
  ON v.injury_id = vi.injury_id;

--SELECT * FROM victim_injury_w_id;

CREATE TABLE victim_injury_w_ethnicity AS
SELECT
  -- all the other nibrs_victim_injury columns you need
  vi.ethnicity_code,
  vi.ethnicity_name,
  -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_id AS v
JOIN public.nibrs_ethnicity AS vi
  ON v.ethnicity_id = vi.ethnicity_id;

--SELECT * FROM victim_injury_w_ethnicity;

CREATE TABLE victim_injury_w_incident AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.agency_id,
    vi.nibrs_month_id,
    vi.incident_date,
    vi.incident_hour,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_ethnicity AS v
JOIN public.nibrs_incident AS vi
  ON v.incident_id = vi.incident_id;

--SELECT * FROM victim_injury_w_incident;

CREATE TABLE victim_injury_w_agencies AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.reporting_type,
	vi.ori,
    vi.ucr_agency_name,
    vi.pub_agency_name,
    vi.state_name,
    vi.population,
    vi.male_officer,
    vi.male_civilian,
    vi.female_officer,
    vi.female_civilian,
    vi.officer_rate,
    vi.employee_rate,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_incident AS v
JOIN public.agencies AS vi
  ON v.agency_id = vi.agency_id;

 CREATE TABLE victim_injury_w_lat_long AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.agency_name,
	vi.latitude,
	vi.longitude,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_agencies AS v
JOIN public.nibrs_agency_locations AS vi
  ON v.ori = vi.ori;

-- SELECT * FROM victim_injury_w_lat_long;

CREATE TABLE victim_injury_w_activity AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.activity_type_code,
	vi.activity_type_name,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_lat_long AS v
JOIN public.nibrs_activity_type AS vi
  ON v.activity_type_id = vi.activity_type_id;

--SELECT * FROM victim_injury_w_activity;

CREATE TABLE victim_injury_w_age AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.age_code,
	vi.age_name,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_activity AS v
JOIN public.nibrs_age AS vi
  ON v.age_id = vi.age_id;

--SELECT * FROM victim_injury_w_age;

CREATE TABLE victim_injury_w_race AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.race_code,
	vi.race_desc,
	vi.sort_order,
	vi.start_year,
	vi.end_year,
	vi.notes,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_age AS v
LEFT OUTER JOIN public.ref_race AS vi
  ON v.race_id = vi.race_id;

--SELECT * FROM victim_injury_w_race;

CREATE TABLE victim_injury_w_circumstances AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.circumstances_id,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_race AS v
LEFT OUTER JOIN public.nibrs_victim_circumstances AS vi
  ON v.victim_id = vi.victim_id;

--SELECT * FROM victim_injury_w_circumstances;

CREATE TABLE victim_injury_w_offender_rel AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.offender_id,
	vi.relationship_id,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_circumstances AS v
LEFT OUTER JOIN public.nibrs_victim_offender_rel AS vi
  ON v.victim_id = vi.victim_id;

--SELECT * FROM victim_injury_w_offender_rel;

CREATE TABLE victim_injury_w_offense AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.offense_id,
	vi.offense_code,
	vi.location_id,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_offender_rel AS v
LEFT OUTER JOIN public.nibrs_offense AS vi
  ON v.incident_id = vi.incident_id;

--SELECT * FROM victim_injury_w_offense;

CREATE TABLE victim_injury_w_offense_type AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.offense_name,
    vi.crime_against,
    vi.ct_flag,
    vi.hc_flag,
    vi.hc_code,
    vi.offense_category_name,
    vi.offense_group,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_offense AS v
LEFT OUTER JOIN public.nibrs_offense_type AS vi
  ON v.offense_code = vi.offense_code;

--SELECT * FROM victim_injury_w_offense_type;

CREATE TABLE victim_injury_w_location AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.location_code,
	vi.location_name,
	
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_offense_type AS v
LEFT OUTER JOIN public.nibrs_location_type AS vi
  ON v.location_id = vi.location_id;

--SELECT * FROM victim_injury_w_location;

CREATE TABLE victim_injury_w_relationship AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.relationship_code,
	vi.relationship_name,
	
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_location AS v
LEFT OUTER JOIN nibrs_relationship AS vi
  ON v.relationship_id = vi.relationship_id;

--SELECT * FROM victim_injury_w_relationship;

CREATE TABLE victim_injury_w_bias_mot AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.bias_id,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_relationship AS v
LEFT OUTER JOIN nibrs_bias_motivation AS vi
  ON v.offense_id = vi.offense_id;

--SELECT * FROM victim_injury_w_bias_mot;

CREATE TABLE victim_injury_w_bias AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.bias_code,
	--vi.bias_category,
	vi.bias_desc,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_bias_mot AS v
LEFT OUTER JOIN nibrs_bias_list AS vi
  ON v.bias_id = vi.bias_id;

--SELECT * FROM victim_injury_w_bias;

CREATE TABLE victim_injury_w_weapon AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.weapon_id,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_bias AS v
LEFT OUTER JOIN nibrs_weapon AS vi
  ON v.offense_id = vi.offense_id;

--SELECT * FROM victim_injury_w_weapon;

CREATE TABLE victim_injury_w_weapon_arrestee AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.arrestee_id,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_weapon AS v
LEFT OUTER JOIN nibrs_arrestee_weapon AS vi
  ON v.weapon_id = vi.weapon_id;

--SELECT * FROM victim_injury_w_weapon_arrestee;

CREATE TABLE MASTER_2021 AS
SELECT
  -- all the other nibrs_victim_injury columns you need
    vi.weapon_name,
	vi.shr_flag,
--   -- ... add more vi.* columns here ...

  -- then the victim columns you need (exclude victim_id)
  v.*
  -- ... add more v.* columns here ...
FROM victim_injury_w_weapon_arrestee AS v
LEFT OUTER JOIN nibrs_weapon_type AS vi
  ON v.weapon_id = vi.weapon_id;

--SELECT * FROM MASTER_20231

ALTER TABLE MASTER_2021
--DROP COLUMN source_file, 
DROP COLUMN resident_status_code, 
DROP COLUMN ethnicity_id,
DROP COLUMN race_id,
DROP COLUMN age_id,
DROP COLUMN activity_type_id, 
DROP COLUMN incident_id,
DROP COLUMN injury_id,
DROP COLUMN injury_code, 
DROP COLUMN ethnicity_code,
DROP COLUMN arrestee_id, 
--DROP COLUMN weapon_code, 
DROP COLUMN weapon_id,
DROP COLUMN bias_code,
DROP COLUMN bias_id,
DROP COLUMN relationship_code, 
DROP COLUMN location_code,
DROP COLUMN offense_code,
DROP COLUMN offender_id, 
DROP COLUMN relationship_id,
DROP COLUMN circumstances_id,
DROP COLUMN race_code,
DROP COLUMN activity_type_code,
DROP COLUMN agency_id,
DROP COLUMN nibrs_month_id,
DROP COLUMN victim_seq_num,
DROP COLUMN victim_id,
--DROP COLUMN offense_id,
DROP COLUMN location_id,
DROP COLUMN offense_id,
DROP COLUMN age_num,
DROP COLUMN start_year,
DROP COLUMN end_year,
DROP COLUMN notes;

--SELECT * FROM MASTER_2021;

DELETE FROM MASTER_2021
WHERE age_code = '00';

--SELECT * FROM MASTER_2021;

DELETE FROM MASTER_2021
WHERE crime_against!='Person';

SELECT * FROM MASTER_2021;

-- copy public.master_2023(
--   weapon_name, shr_flag, bias_desc, relationship_name, location_name,
--   offense_name, crime_against, ct_flag, hc_flag, hc_code, offense_category_name,
--   offense_group, race_desc, sort_order, /* start_year removed if it doesn't exist */
--   /* end_year removed if it doesn't exist */
--   age_code, age_name, activity_type_name, reporting_type, ucr_agency_name,
--   pub_agency_name, state_name, population, male_officer, male_civilian, female_officer,
--   female_civilian, officer_rate, employee_rate, incident_date, incident_hour,
--   ethnicity_name, injury_name, data_year, sex_code, age_range_low_num, age_range_high_num
-- )
-- TO '/Users/ritaherfi/Desktop/BDA 600 Data/Raw/2021/MASTER_2021.csv'
-- WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '''');
