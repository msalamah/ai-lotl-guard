# Split Report

## Train
Rows: 126 | Groups: 90 | Malicious: 80 | Benign: 46 | Malicious ratio: 0.63

## Validation
Rows: 28 | Groups: 23 | Malicious: 17 | Benign: 11 | Malicious ratio: 0.61

## Test
Rows: 50 | Groups: 44 | Malicious: 28 | Benign: 22 | Malicious ratio: 0.56

## Top group_key clusters
- c:\windows\explorer.exe::c:\windows\explorer.exe: 7 rows
- c:\program files\7-zip\7zfm.exe::c:\program files\<NUM>-zip\7zfm.exe: 4 rows
- c:\windows\system32\cmd.exe::cmdkey /list: 4 rows
- c:\program files\git\bin\bash.exe::c:\program files\git\bin\bash.exe: 4 rows
- c:\program files\wireshark\wireshark.exe::c:\program files\wireshark\wireshark.exe: 4 rows
- c:\windows\system32\notepad.exe::notepad.exe c:\users\jsmith\documents\notes.txt: 3 rows
- c:\program files\mozilla firefox\firefox.exe::c:\program files\mozilla firefox\firefox.exe: 3 rows
- c:\windows\system32\reg.exe::reg query hklm\software\microsoft\windows nt\currentversion\winlogon: 3 rows
- c:\windows\system32\findstr.exe::findstr /spin "password" *.*: 3 rows
- c:\windows\system32\cmd.exe::cmd.exe /c copy c:\users\public\documents\sensitive_data.txt c:\temp\data.cab: 3 rows
