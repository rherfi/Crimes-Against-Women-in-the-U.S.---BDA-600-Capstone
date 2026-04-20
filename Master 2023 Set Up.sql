-- This file is used to setup the database tables and load the NIBRS
-- code lookup tables. It only needs to be run once before you load
-- any data tables using postgres_load.sql

SET statement_timeout = 0;
SET lock_timeout = 0;
-- SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';

CREATE TABLE agencies (
    yearly_agency_id integer,
    agency_id integer,
    data_year integer,
    ori character varying(25),
    legacy_ori character varying(25),
    covered_by_legacy_ori character varying(25),
    direct_contributor_flag character varying(1),
    dormant_flag character varying(1),
    dormant_year integer,
    reporting_type character varying(1),
    ucr_agency_name character varying(100),
    ncic_agency_name character varying(100),
    pub_agency_name character varying(100),
    pub_agency_unit character varying(100),
    agency_status character varying(1),
    state_id integer,
    state_name character varying(100),
    state_abbr character varying(2),
    state_postal_abbr character varying(2),
    division_code integer,
    division_name character varying(100),
    region_code integer,
    region_name character varying(100),
    region_desc character varying(100),
    agency_type_name character varying(100),
    population integer,
    submitting_agency_id integer,
    sai character varying(25),
    submitting_agency_name character varying(200),
    suburban_area_flag character varying(1),
    population_group_id integer,
    population_group_code character varying(2),
    population_group_desc character varying(100),
    parent_pop_group_code integer,
    parent_pop_group_desc character varying(100),
    mip_flag character varying(1),
    pop_sort_order integer,
    summary_rape_def character varying(1),
    pe_reported_flag character varying(1),
    male_officer integer,
    male_civilian integer,
    male_total integer,
    female_officer integer,
    female_civilian integer,
    female_total integer,
    officer_rate decimal,
    employee_rate decimal,
    nibrs_cert_date date,
    nibrs_start_date date,
    nibrs_leoka_start_date date,
    nibrs_ct_start_date date,
    nibrs_multi_bias_start_date date,
    nibrs_off_eth_start_date date,
    covered_flag character varying(1),
    county_name character varying(100),
    msa_name character varying(100),
    publishable_flag character varying(1),
    participated character varying(1),
    nibrs_participated character varying(1)
);

CREATE TABLE nibrs_agency_locations (
    year                 integer NOT NULL,
    ori character varying(25) NOT NULL,
    agency_name          character varying(100) NOT NULL,
    agency_type          character varying(100),
    county               character varying(100),
    state                char(2) NOT NULL,
    latitude             double precision,
    longitude            double precision,
    nibrs_participation  boolean NOT NULL,
    -- -- Optional extras if present in your master file:
    -- is_nibrs             boolean,
    -- nibrs_start_date     date,
    CONSTRAINT nibrs_agency_locations_pk PRIMARY KEY (year, ori),
    CONSTRAINT nibrs_state_len CHECK (char_length(state) = 2),
    CONSTRAINT nibrs_lat_range CHECK (latitude  IS NULL OR (latitude  >= -90  AND latitude  <= 90)),
    CONSTRAINT nibrs_lon_range CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180))
);

-- -- Useful index for spatial-ish queries (non-PostGIS)
-- CREATE INDEX nibrs_agency_locations_latlon_idx
--   ON nibrs_agency_locations (latitude, longitude);

-- If you have PostGIS installed and want true geometry points:
-- SELECT AddGeometryColumn('public','nibrs_agency_locations','geom',4326,'POINT',2);
-- UPDATE nibrs_agency_locations SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);
-- CREATE INDEX nibrs_agency_locations_geom_gix ON nibrs_agency_locations USING GIST (geom);


CREATE TABLE nibrs_activity_type (
activity_type_id smallint NOT NULL,
activity_type_code character(2),
activity_type_name character varying(100)
);

CREATE TABLE nibrs_age (
age_id smallint NOT NULL,
age_code character(2),
age_name character varying(100)
);


CREATE TABLE nibrs_bias_list (
bias_id smallint NOT NULL,
bias_code character(2),
bias_name character varying(100),
bias_desc character varying(100)
);

CREATE TABLE nibrs_location_type (
    location_id bigint NOT NULL,
    location_code character(2),
    location_name character varying(100)
);

CREATE TABLE nibrs_offense_type (
    offense_code character varying(5) NOT NULL,
    offense_name character varying(100),
    crime_against character varying(100),
    ct_flag character(1),
    hc_flag character(1),
    hc_code character varying(5),
    offense_category_name character varying(100),
    offense_group character(5)
);

CREATE TABLE nibrs_circumstances (
    circumstances_id smallint NOT NULL,
    circumstances_type varchar,
    circumstances_code smallint,
    circumstances_name character varying(100)
);


CREATE TABLE nibrs_ethnicity (
    ethnicity_id smallint NOT NULL,
    ethnicity_code character(1),
    ethnicity_name character varying(100)
);

CREATE TABLE nibrs_injury (
injury_id smallint NOT NULL,
injury_code character(1),
injury_name character varying(100)
);

CREATE TABLE nibrs_relationship (
relationship_id smallint NOT NULL,
relationship_code character(2),
relationship_name character varying(100)
);

CREATE TABLE nibrs_weapon_type (
weapon_id smallint NOT NULL,
weapon_code character varying(3),
weapon_name character varying(100),
shr_flag character(1)
);

CREATE TABLE ref_race (
race_id smallint NOT NULL,
race_code character varying(2) NOT NULL,
race_desc character varying(100) NOT NULL,
sort_order smallint,
start_year smallint,
end_year smallint,
notes character varying(1000)
);


--
-- Main NIBRS tables
--


CREATE TABLE nibrs_arrestee_weapon (
data_year int,
arrestee_id bigint NOT NULL,
nibrs_arrestee_weapon_id bigint,
weapon_id smallint NOT NULL
);

CREATE TABLE nibrs_bias_motivation (
data_year int,
bias_id smallint NOT NULL,
offense_id bigint NOT NULL
);

CREATE TABLE nibrs_incident (
	data_year int,
    agency_id bigint NOT NULL,
    incident_id bigint NOT NULL,
    nibrs_month_id bigint NOT NULL,
    cargo_theft_flag character(1),
    submission_date timestamp without time zone,
    incident_date timestamp without time zone,
    report_date_flag character(1),
    incident_hour smallint,
    cleared_except_id smallint NOT NULL,
    cleared_except_date timestamp without time zone,
    incident_status character varying(100), --smallint,
    data_home character(1),
    orig_format character(1),
    did bigint
);

COMMENT ON COLUMN nibrs_incident.orig_format IS 'This is the format the report was in when it was first submitted to the system.  F for Flat File, W for Web Form, U for IEPDXML Upload, S for IEPDXML Service, B for BPEL, N for null or unavailable.';

CREATE TABLE nibrs_offense (
data_year int,
offense_id bigint NOT NULL,
incident_id bigint NOT NULL,
offense_code character varying(5) NOT NULL,
attempt_complete_flag character(1),
location_id bigint NOT NULL,
num_premises_entered smallint,
method_entry_code character(1)
);

CREATE TABLE nibrs_victim (
    data_year int,
    victim_id bigint NOT NULL,
    incident_id bigint NOT NULL,
    victim_seq_num smallint,
    victim_type_id smallint NOT NULL,
    assignment_type_id smallint,
    activity_type_id smallint,
    outside_agency_id bigint,
    age_id smallint,
    age_num varchar,
    sex_code character(1),
    race_id smallint,
    ethnicity_id smallint,
    resident_status_code character(1),
    age_range_low_num smallint,
    age_range_high_num smallint
);

CREATE TABLE nibrs_victim_circumstances (
    data_year int,
    victim_id bigint NOT NULL,
    circumstances_id smallint NOT NULL,
    justifiable_force_id smallint
);

CREATE TABLE nibrs_victim_injury (
    data_year int,
    victim_id bigint NOT NULL,
    injury_id smallint NOT NULL
);

CREATE TABLE nibrs_victim_offender_rel (
    data_year int,
    victim_id bigint NOT NULL,
    offender_id bigint NOT NULL,
    relationship_id smallint NOT NULL,
    nibrs_victim_offender_id bigint
);

CREATE TABLE nibrs_weapon (
    data_year int,
    weapon_id smallint NOT NULL,
    offense_id bigint NOT NULL,
    nibrs_weapon_id bigint
);

--Create PKs and FKs
--PKS

  ALTER TABLE ONLY PUBLIC.NIBRS_RELATIONSHIP ADD CONSTRAINT NIBRS_RELATIONSHIP_PK PRIMARY KEY (RELATIONSHIP_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_ETHNICITY ADD CONSTRAINT NIBRS_ETH_PK PRIMARY KEY (ETHNICITY_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_OFFENSE_TYPE ADD CONSTRAINT NIBRS_OFFENSE_TYPE_PK PRIMARY KEY (offense_code);

  ALTER TABLE ONLY PUBLIC.NIBRS_INCIDENT ADD CONSTRAINT NIBRS_INCIDENT_PK PRIMARY KEY (INCIDENT_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_BIAS_MOTIVATION ADD CONSTRAINT NIBRS_BIAS_MOTIVATION_PK PRIMARY KEY (BIAS_ID, OFFENSE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_INJURY ADD CONSTRAINT NIBRS_INJURY_PK PRIMARY KEY (INJURY_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_CIRCUMSTANCES ADD CONSTRAINT NIBRS_CIRCUMSTANCES_PK PRIMARY KEY (CIRCUMSTANCES_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_ACTIVITY_TYPE ADD CONSTRAINT NIBRS_ACTIVITY_TYPE_PK PRIMARY KEY (ACTIVITY_TYPE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_WEAPON_TYPE ADD CONSTRAINT NIBRS_WEAPON_TYPE_PK PRIMARY KEY (WEAPON_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_BIAS_LIST ADD CONSTRAINT NIBRS_BIAS_LIST_PK PRIMARY KEY (BIAS_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_INJURY ADD CONSTRAINT NIBRS_VICTIM_INJURY_PK PRIMARY KEY (VICTIM_ID, INJURY_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_OFFENSE ADD CONSTRAINT NIBRS_OFFENSE_PK PRIMARY KEY (OFFENSE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_OFFENDER_REL ADD CONSTRAINT NIBRS_VICTIM_OFFENDER_REL_PK PRIMARY KEY (victim_id, offender_id);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM ADD CONSTRAINT NIBRS_VICTIM_PK PRIMARY KEY (VICTIM_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_AGE ADD CONSTRAINT NIBRS_AGE_PK PRIMARY KEY (AGE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_CIRCUMSTANCES ADD CONSTRAINT NIBRS_VICTIM_CIRCUMSTANCES_PK PRIMARY KEY (VICTIM_ID, CIRCUMSTANCES_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_ARRESTEE_WEAPON ADD CONSTRAINT NIBRS_ARRESTEE_WEAPON_PK PRIMARY KEY (arrestee_id, weapon_id);

  ALTER TABLE ONLY PUBLIC.NIBRS_LOCATION_TYPE ADD CONSTRAINT NIBRS_LOCATION_TYPE_PK PRIMARY KEY (LOCATION_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_WEAPON ADD CONSTRAINT NIBRS_WEAPON_PK PRIMARY KEY (weapon_id, offense_id);

  ALTER TABLE ONLY PUBLIC.REF_RACE ADD CONSTRAINT REF_RACE_PK PRIMARY KEY (RACE_ID);

  ALTER TABLE ONLY PUBLIC.AGENCIES ADD CONSTRAINT AGENCIES_PK PRIMARY KEY (agency_id);

  -- FKs
  ALTER TABLE ONLY PUBLIC.NIBRS_ARRESTEE_WEAPON ADD CONSTRAINT NIBRS_ARREST_WEAP_TYPE_FK FOREIGN KEY (WEAPON_ID)
	  REFERENCES PUBLIC.NIBRS_WEAPON_TYPE (WEAPON_ID);
  
  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_INJURY ADD CONSTRAINT NIBRS_VIC_INJURY_NIBRS_INJ_FK FOREIGN KEY (INJURY_ID)
	  REFERENCES PUBLIC.NIBRS_INJURY (INJURY_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_INJURY ADD CONSTRAINT NIBRS_VIC_INJURY_NIBRS_VIC_FK FOREIGN KEY (VICTIM_ID)
	  REFERENCES PUBLIC.NIBRS_VICTIM (VICTIM_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_WEAPON ADD CONSTRAINT NIBRS_WEAP_WEAP_TYPE_FK FOREIGN KEY (WEAPON_ID)
	  REFERENCES PUBLIC.NIBRS_WEAPON_TYPE (WEAPON_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_WEAPON ADD CONSTRAINT NIBRS_WEAP_OFF_FK FOREIGN KEY (OFFENSE_ID)
	  REFERENCES PUBLIC.NIBRS_OFFENSE (OFFENSE_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_CIRCUMSTANCES ADD CONSTRAINT NIBRS_VIC_CIRC_NIBRS_VIC_FK FOREIGN KEY (VICTIM_ID)
	  REFERENCES PUBLIC.NIBRS_VICTIM (VICTIM_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_CIRCUMSTANCES ADD CONSTRAINT NIBRS_VIC_CIRC_NIBRS_CIRC_FK FOREIGN KEY (CIRCUMSTANCES_ID)
	  REFERENCES PUBLIC.NIBRS_CIRCUMSTANCES (CIRCUMSTANCES_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_OFFENSE ADD CONSTRAINT NIBRS_OFFENSE_INC_FK1 FOREIGN KEY (INCIDENT_ID)
	  REFERENCES PUBLIC.NIBRS_INCIDENT (INCIDENT_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_OFFENSE ADD CONSTRAINT NIBRS_OFFENSE_LOC_TYPE_FK FOREIGN KEY (LOCATION_ID)
	  REFERENCES PUBLIC.NIBRS_LOCATION_TYPE (LOCATION_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_OFFENSE ADD CONSTRAINT NIBRS_OFFENSE_OFF_TYPE_FK FOREIGN KEY (offense_code)
	  REFERENCES PUBLIC.NIBRS_OFFENSE_TYPE (offense_code);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_OFFENDER_REL ADD CONSTRAINT NIBRS_VICTIM_OFF_REL_VIC_FK FOREIGN KEY (VICTIM_ID)
	  REFERENCES PUBLIC.NIBRS_VICTIM (VICTIM_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM_OFFENDER_REL ADD CONSTRAINT NIBRS_VICTIM_OFF_REL_REL_FK FOREIGN KEY (RELATIONSHIP_ID)
	  REFERENCES PUBLIC.NIBRS_RELATIONSHIP (RELATIONSHIP_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM ADD CONSTRAINT NIBRS_VICTIM_ACT_TYPE_FK FOREIGN KEY (ACTIVITY_TYPE_ID)
	  REFERENCES PUBLIC.NIBRS_ACTIVITY_TYPE (ACTIVITY_TYPE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM ADD CONSTRAINT NIBRS_VICTIM_AGE_FK FOREIGN KEY (AGE_ID)
	  REFERENCES PUBLIC.NIBRS_AGE (AGE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM ADD CONSTRAINT NIBRS_VICTIM_ETHNICITY_FK FOREIGN KEY (ETHNICITY_ID)
	  REFERENCES PUBLIC.NIBRS_ETHNICITY (ETHNICITY_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM ADD CONSTRAINT NIBRS_VICTIM_RACE_FK FOREIGN KEY (RACE_ID)
	  REFERENCES PUBLIC.REF_RACE (RACE_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_VICTIM ADD CONSTRAINT NIBRS_VICTIM_INC_FK FOREIGN KEY (INCIDENT_ID)
	  REFERENCES PUBLIC.NIBRS_INCIDENT (INCIDENT_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_INCIDENT ADD CONSTRAINT NIBRS_INCIDENT_AGENCY_FK FOREIGN KEY (AGENCY_ID)
	  REFERENCES PUBLIC.AGENCIES (AGENCY_ID);

  ALTER TABLE ONLY PUBLIC.NIBRS_BIAS_MOTIVATION ADD CONSTRAINT NIBRS_BIAS_MOT_OFFENSE_FK FOREIGN KEY (OFFENSE_ID)
	  REFERENCES PUBLIC.NIBRS_OFFENSE (OFFENSE_ID) ON DELETE CASCADE;

  ALTER TABLE ONLY PUBLIC.NIBRS_BIAS_MOTIVATION ADD CONSTRAINT NIBRS_BIAS_MOT_LIST_FK FOREIGN KEY (BIAS_ID)
	  REFERENCES PUBLIC.NIBRS_BIAS_LIST (BIAS_ID);
