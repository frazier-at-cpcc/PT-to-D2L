# Brightspace Packet Tracer Grade Processor

A Streamlit application that processes Packet Tracer activity grades from CSV files and updates Brightspace gradebooks.

## Features

- **File Upload**: Upload both Brightspace CSV and Grade Details CSV files
- **Activity Selection**: Choose which Packet Tracer activity to update from a dropdown
- **Student Matching**: Automatically matches students between the two CSV files based on names
- **Grade Processing**: Updates the selected Packet Tracer activity with grades from Grade Details
- **Auto-Zero Feature**: Automatically assigns zero grades to students who didn't submit
- **CSV Export**: Download the updated Brightspace CSV file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Usage

### Step 1: Prepare Your Files
- **Brightspace CSV**: Export your gradebook from Brightspace
- **Grade Details CSV**: Export grade details from Packet Tracer activity with student submissions

### Step 2: Upload Files
1. Upload your Brightspace CSV file using the first file uploader
2. Upload your Grade Details CSV file using the second file uploader

### Step 3: Select Activity
- Choose the Packet Tracer activity you want to update from the dropdown menu
- The app will show all available Packet Tracer activities found in your Brightspace CSV

### Step 4: Review Matches
- The app will automatically match students between the two files
- Review the matching results to ensure accuracy
- Check the debug information if needed to troubleshoot any issues

### Step 5: Update Grades
- Click "Update Grades" to apply the grades from Grade Details to the Brightspace CSV
- Optionally click "Auto-load Zeros for Missing Grades" to assign zeros to non-submitters

### Step 6: Download Updated File
- Review the preview of updated grades
- Click "Download Updated Brightspace CSV" to get your updated file

## File Format Requirements

### Brightspace CSV
- Must contain columns: `OrgDefinedId`, `Last Name`, `First Name`, `Email`
- Must contain Packet Tracer activity columns with "Packet Tracer" and "Points Grade" in the name

### Grade Details CSV
- Must contain a column with filenames (accepts variations like "Filename", "File Name", etc.)
- Must contain a total percentage column (accepts variations like "Total Percentage", "Total_Percentage", etc.)
- Filenames should follow the pattern: `[ID] - [Student Name] - [Date/Time].pka`

## Troubleshooting

### No Matching Students Found
- Check that student names in Grade Details filenames match those in Brightspace
- Verify the filename format in Grade Details CSV
- Use the debug information to inspect column names and data

### Column Name Errors
- The app automatically handles common variations in column names
- Check the debug section to see the exact column names in your files
- Ensure your CSV files are properly formatted

### Missing Grades
- Use the "Auto-load Zeros" feature to assign zeros to missing submissions
- Review the preview before downloading to ensure accuracy

## Support

If you encounter issues:
1. Check the debug information in the expandable section
2. Verify your CSV file formats match the requirements
3. Ensure student names are consistent between files