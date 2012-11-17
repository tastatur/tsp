set term png
set output 'async.png'

set title "Actual vs Computed Path"
set xlabel "Postion x co-ordinate"
set ylabel "Position y co-ordinate"
set grid
plot "TSPActual.dat" notitle with linespoints

# eof

