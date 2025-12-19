# Data Overview

Total rows: 204
Malicious rows: 125
Benign rows: 79
Malicious ratio: 61.27%

## Top SourceImage processes
- c:\windows\system32\cmd.exe: 68
- c:\windows\system32\windowspowershell\v1.0\powershell.exe: 20
- c:\windows\system32\findstr.exe: 8
- c:\windows\explorer.exe: 7
- c:\windows\system32\reg.exe: 7
- c:\windows\system32\powershell.exe: 6
- c:\windows\system32\certutil.exe: 5
- c:\program files\microsoft vs code\code.exe: 4
- c:\program files\git\bin\bash.exe: 4
- c:\program files\mozilla firefox\firefox.exe: 4

## Largest group_key clusters
- c:\windows\explorer.exe::c:\windows\explorer.exe: 7 events
- c:\program files\7-zip\7zfm.exe::c:\program files\<NUM>-zip\7zfm.exe: 4 events
- c:\windows\system32\cmd.exe::cmdkey /list: 4 events
- c:\program files\git\bin\bash.exe::c:\program files\git\bin\bash.exe: 4 events
- c:\program files\wireshark\wireshark.exe::c:\program files\wireshark\wireshark.exe: 4 events
- c:\windows\system32\notepad.exe::notepad.exe c:\users\jsmith\documents\notes.txt: 3 events
- c:\program files\mozilla firefox\firefox.exe::c:\program files\mozilla firefox\firefox.exe: 3 events
- c:\windows\system32\reg.exe::reg query hklm\software\microsoft\windows nt\currentversion\winlogon: 3 events
- c:\windows\system32\findstr.exe::findstr /spin "password" *.*: 3 events
- c:\windows\system32\cmd.exe::cmd.exe /c copy c:\users\public\documents\sensitive_data.txt c:\temp\data.cab: 3 events
- c:\windows\system32\cmd.exe::cmd.exe /c makecab c:\users\pbeesly\documents\projectplans.docx c:\programdata\temp.cab: 3 events

## Insight summary
- Claude vs `_label`: ~95% agreement; disagreements skew malicious where humans marked benign (likely aggressive detection on enumeration/copy commands).
- Malicious class dominates overall (~61%) and within top LOLBins (cmd.exe, powershell.exe, reg.exe).
- Encoded commands and bulk copy/makecab operations cluster into large group_keys; these need robust feature coverage to avoid leakage.
- Benign clusters primarily office software (`WINWORD.EXE`, `Code.exe`) and admin tools (`7zFM.exe`, `Wireshark.exe`)—useful negatives for models.
