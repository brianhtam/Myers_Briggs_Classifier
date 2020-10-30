CREATE DATABASE myers_briggs;

\connect myers_briggs;

CREATE TABLE twitter_origin(
  type TEXT,
  posts TEXT
);

\copy twitter_origin FROM 'data/mbti_1.csv' DELIMITER ',' CSV HEADER;

