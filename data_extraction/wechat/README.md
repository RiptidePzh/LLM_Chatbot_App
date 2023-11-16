## Extract WeChat Data
Guide for getting Encrption Key to WeChat Database (for Mac)

#### step1: Grant lldb access to attach process
* Shutdown your device
* Get your Mac in Recovery Mode by holding `Command+R` on older models of Mac or `power` button on later models
* Open the **Terminal** in the **Utilities Menu**, enter `csrutil disable; reboot`

#### step2: Catch the Encryption KEY
* Open WeChat-pc on your Mac
* In **Terminal**, enter `lldb -p $(pgrep WeChat)` to attach the WeChat process to the debugger
  
  In **（lldb）shell**, enter `br set -n sqlite3_key` to set breakup points where sqlite3_key is called
  
  In **（lldb）shell**, enter `c` to continue the process
* Open WeChat-pc for a little while, could see the window freezes, meaning that the debugger has captured the sqlite3_key breakpoint
* In **（lldb）shell**, enter `memory read --size 1 --format x --count 32 $x1` to read the db encrytion key loaded in the memory
  
  *Note: If this key doesn't work, and you could see `$rsi` (Register Source Index) thread in output of `memory read`, try replacing `$x1` with that instead*

#### step3: Paste the output source key to "source" in extract_chat.py
