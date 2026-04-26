#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -18.45*x up 19.31*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7 ambient 0.4 diffuse 0.55}
#declare pale = finish {ambient 0.9 diffuse 0.30 roughness 0.001 specular 0.2 }
#declare intermediate = finish {ambient 0.4 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.2 diffuse 0.80 phong 0.25 phong_size 10.0 specular 0.2 roughness 0.1}
#declare jmol = finish {ambient 0.4 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.2 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.4 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.4 diffuse 0.35 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.3 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

cylinder {< -5.89,  -3.33, -27.07>, <  2.77,  -3.33, -27.07>, Rcell pigment {Black}}
cylinder {< -1.56,   4.16, -27.07>, <  7.10,   4.16, -27.07>, Rcell pigment {Black}}
cylinder {< -1.56,   4.16,   0.00>, <  7.10,   4.16,   0.00>, Rcell pigment {Black}}
cylinder {< -5.89,  -3.33,   0.00>, <  2.77,  -3.33,   0.00>, Rcell pigment {Black}}
cylinder {< -5.89,  -3.33, -27.07>, < -1.56,   4.16, -27.07>, Rcell pigment {Black}}
cylinder {<  2.77,  -3.33, -27.07>, <  7.10,   4.16, -27.07>, Rcell pigment {Black}}
cylinder {<  2.77,  -3.33,   0.00>, <  7.10,   4.16,   0.00>, Rcell pigment {Black}}
cylinder {< -5.89,  -3.33,   0.00>, < -1.56,   4.16,   0.00>, Rcell pigment {Black}}
cylinder {< -5.89,  -3.33, -27.07>, < -5.89,  -3.33,   0.00>, Rcell pigment {Black}}
cylinder {<  2.77,  -3.33, -27.07>, <  2.77,  -3.33,   0.00>, Rcell pigment {Black}}
cylinder {<  7.10,   4.16, -27.07>, <  7.10,   4.16,   0.00>, Rcell pigment {Black}}
cylinder {< -1.56,   4.16, -27.07>, < -1.56,   4.16,   0.00>, Rcell pigment {Black}}
atom(< -5.89,  -3.33, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #0
atom(< -3.00,  -3.33, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #1
atom(< -0.12,  -3.33, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #2
atom(< -4.44,  -0.83, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #3
atom(< -1.56,  -0.83, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #4
atom(<  1.33,  -0.83, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #5
atom(< -3.00,   1.67, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #6
atom(< -0.12,   1.67, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #7
atom(<  2.77,   1.67, -17.07>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #8
atom(< -4.44,  -2.50, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #9
atom(< -1.56,  -2.50, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #10
atom(<  1.33,  -2.50, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #11
atom(< -3.00,   0.00, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #12
atom(< -0.12,   0.00, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #13
atom(<  2.77,   0.00, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #14
atom(< -1.56,   2.50, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #15
atom(<  1.33,   2.50, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #16
atom(<  4.21,   2.50, -14.71>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #17
atom(< -5.89,  -1.67, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #18
atom(< -3.00,  -1.67, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #19
atom(< -0.12,  -1.67, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #20
atom(< -4.44,   0.83, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #21
atom(< -1.56,   0.83, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #22
atom(<  1.33,   0.83, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #23
atom(< -3.00,   3.33, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #24
atom(< -0.12,   3.33, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #25
atom(<  2.77,   3.33, -12.36>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #26
atom(< -5.89,  -3.33, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #27
atom(< -3.00,  -3.33, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #28
atom(< -0.12,  -3.33, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #29
atom(< -4.44,  -0.83, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #30
atom(< -1.56,  -0.83, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #31
atom(<  1.33,  -0.83, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #32
atom(< -3.00,   1.67, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #33
atom(< -0.12,   1.67, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #34
atom(<  2.77,   1.67, -10.00>, 1.21, rgb <1.00, 0.82, 0.13>, 0.0, ase2) // #35

// no constraints
