set term png
set output 'async.png'

set title "Actual vs Computed Path"
set xlabel "Postion x co-ordinate"
set ylabel "Position y co-ordinate"
set grid
plot "finalOutput.dat" notitle with linespoints

# eof

