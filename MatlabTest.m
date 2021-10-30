u = readmatrix('ini_u.csv');
N = readmatrix('ini_N.csv');

d = -N\u;

% recalculating A_e

A_e = [];
rotmat = readmatrix()