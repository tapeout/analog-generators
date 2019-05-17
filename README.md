# analog-generators
BAG scripts for analog design, simulation, and layout.
## Obey the NDA
Do not under **any** circumstances put proprietary information in this repository!
## Best Practices
* Naming convention: `FUNCTION_topologyName_yourName.py`
  * Examples of `FUNCTION`: OTA, TIA, OPAMP, REFERENCE, DAC
  * Examples of `topologyName`: diffAmpN, diffAmpP, foldedCascodeP, foldedCascodeN, bandgap, constantGm
  * Example of a file name: OTA_diffAmpN_lydiaLee.py
* At the top of each generator, include the following:
  * Name
  * Semester/Year: Last updated or created
  
## Process Characterization
0. Source the .cshrc (with csh)
    * `source .cshrc`
1. On the cluster, run Cadence Virtuoso
    * `virtuoso &`
    * The ampersand (&) will allow you to continue using the terminal normally even when Virtuoso is running
2. Add the necessary cds.libs to your default cds.lib
    * Add this to your cds.lib: `INCLUDE $BAG_WORK_DIR/cds.lib.core`  
3. In the Virtuoso CDF window, start a BAG server:
    * `load("start_bag.il")`
    * Wait until the Virtuoso window indicates a BAG server has been started
4. Create a .yaml file to configure your characterization settings
    * Templates have been provided in specs_mos_char/nch_w0d5_100nm.yaml and specs_mos_char/pch_w0d5_100nm.yaml
    * The templates characterize using schematic only (view_name: 'schematic')
5. Copy scripts_char/mos_char.py and modify the copy to point to the .yaml file you'd like to use for characterization settings
6. Start BAG
    * `./start_bag.sh`
7. Run the script you created in (4)
    * `run -i ./path-to-your-script/your-script.py`
    * The '-i' flag isn't mandatory, but it's useful for debugging
8. Wait
    * Depending on what you're running, this could take a while
    * Noise simulations in particular take a long time
    
## Running Your Own Scripts
0. Source the .cshrc (with csh)
    * `source .cshrc`
1. If you need to interface with Virtuoso:
    1. On the cluster, run Cadence Virtuoso
        * `virtuoso &`
        * The ampersand (&) will allow you to continue using the terminal normally even when Virtuoso is running
    2. In the Virtuoso CDF window, start a BAG server:
        * `load("start_bag.il")`
        * Wait until the Virtuoso window indicates a BAG server has been started
2. Start BAG in Terminal
    * `./start_bag.sh`
3. Run the script you created
    * `run -i ./path-to-your-script/your-script.py`
    * The '-i' flag isn't mandatory, but it's useful for debugging
