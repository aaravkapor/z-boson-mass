# Z-boson mass: histogram peak vs Gaussian fit

Compares tallest-bin and Gaussian-fit estimates of the Z mass on simulated
and real CMS Open Data (CERN Open Data record 700, Run2010B dimuons).

Install: python3 -m pip install numpy scipy pandas matplotlib

Run, in this order:
1. python3 zmc.py                      -> mc_results.json (Tables 1 and 3)
2. python3 nscan.py                    -> nscan.json, fig2.png (Table 1 / Figure 1)
3. python3 figs.py                     -> fig1-4.png, binscan.json (Table 2)
4. python3 analyze_cms_dimuon.py MuRun2010B.csv
                                       -> real_data_results.json, real_data_fit.png (Table 4)

Data: download MuRun2010B.csv from opendata.cern.ch/record/700
Seeds are fixed, so the results should reproduce.
