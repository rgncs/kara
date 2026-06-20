---
name: data-to-spreadsheet
description: Turn data from the conversation into a real .xlsx or .csv file.
triggers: (put|save|export|turn|make).*(spreadsheet|excel|\.xlsx|csv|table), spreadsheet of, excel (file|sheet) (of|with|for)
---
When I ask you to put data into a spreadsheet or Excel file:

1. The data is whatever we were just discussing in THIS conversation (a list,
   results, numbers) — never my stored personal background. If it's unclear what
   data I mean, ask one short question before writing anything.
2. For a real `.xlsx`, write a short Python script with `write_file` and run it
   with `run_command` (python resolves to Kara's venv, which has openpyxl):
   - Create a workbook, add a header row, then the data rows.
   - Bold the header, set sensible column widths, freeze the header row.
   - Save into the workspace and report the filename.
3. For a plain `.csv`, `write_file` is enough — no script needed.
4. Keep columns meaningful (name them), and don't invent data to fill cells; if a
   value is unknown, leave it blank.
5. Confirm what you created in one line (filename + what's in it). The shell step
   still goes through the approval gate.
