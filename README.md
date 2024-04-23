# Utility for xcal gui parsing
Using pywinauto to parse and control XCAL5 window.

## Run
Executable script format: parse_*.py

## TODO
* Test and list what XCAL reports are useful.
* Parse them using script.

## MISC
List useful xcal report item.

### 5GNR
- MAC PDSCH -> RB num, TB size, per slot
- MAC DCI info -> RB num, per slot

MAC: DCI info, UCI info, PDSCH status
RLC: UL Status PDU, Status PDU(?)
PDCP: DL Data PDU

### LTE:
TODO: Can't find per slot RB report in LTE. Unexpected.
- RLC: DL AM All PDU: TB size per slot.
