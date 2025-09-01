import streamlit as st
import pandas as pd
import re
from io import StringIO
import numpy as np

def extract_student_name_from_filename(filename):
    """Extract student name from the filename in grades-details CSV"""
    if pd.isna(filename) or filename == '':
        return None
    
    # Convert to string in case it's not
    filename = str(filename)
    
    # Pattern to extract name from filename like "117493-618463 - Sayvon Cook-Mccullough - Aug 29, 2025 208 PM.pka"
    patterns = [
        r'\d+-\d+ - (.+?) - \w+ \d+, \d+ .+\.pka',  # Full pattern
        r'\d+-\d+ - (.+?) - .+\.pka',  # Simplified pattern
        r'- (.+?) - \w+ \d+',  # Even more simplified
        r'- (.+?) -',  # Just between dashes
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1).strip()
    
    return None

def transform_grades_csv(grades_df):
    """Transform the grades CSV to create a standardized mapping table"""
    transformed_data = []
    
    # Debug: Print what we're working with
    st.write("**🔍 Transformation Debug:**")
    st.write(f"DataFrame shape: {grades_df.shape}")
    st.write(f"Columns: {list(grades_df.columns)}")
    
    for idx, row in grades_df.iterrows():
        st.write(f"\n**Row {idx}:**")
        
        # Try to find filename in any column
        filename = None
        for col_idx, col in enumerate(grades_df.columns):
            cell_value = str(row[col])
            st.write(f"  Column {col_idx} ('{col}'): '{cell_value}'")
            
            # Check for .pka files with more flexible criteria
            if '.pka' in cell_value.lower():
                st.write(f"    ✅ Found .pka file!")
                filename = cell_value
                break
        
        if filename:
            st.write(f"  📄 Using filename: {filename}")
            student_name = extract_student_name_from_filename(filename)
            st.write(f"  👤 Extracted name: {student_name}")
            
            if student_name:
                # Split name into first and last
                name_parts = student_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = ' '.join(name_parts[1:])  # Handle multiple last names
                    
                    # Get the grade (try multiple column names)
                    grade = None
                    grade_cols = ['Total Percentage', 'Total_Percentage', 'total percentage', 'Grade', 'Score']
                    for col in grade_cols:
                        if col in grades_df.columns and pd.notna(row[col]):
                            grade = row[col]
                            st.write(f"  📊 Found grade in '{col}': {grade}")
                            break
                    
                    transformed_data.append({
                        'Original_Index': idx,
                        'Full_Name': student_name,
                        'First_Name': first_name,
                        'Last_Name': last_name,
                        'Filename': filename,
                        'Grade': grade,
                        'Grade_Numeric': float(grade) if pd.notna(grade) and str(grade).replace('.','').replace(',','').isdigit() else None
                    })
                    st.write(f"  ✅ Added to transformed data")
                else:
                    st.write(f"  ❌ Could not split name into parts")
            else:
                st.write(f"  ❌ Could not extract student name from filename")
        else:
            st.write(f"  ❌ No .pka filename found in this row")
        
        # Stop after a few rows for debugging
        if idx >= 2:
            st.write("(Showing first 3 rows only...)")
            break
    
    return pd.DataFrame(transformed_data)

def extract_packet_tracer_columns(df):
    """Extract Packet Tracer activity columns from Brightspace CSV"""
    pt_columns = []
    for col in df.columns:
        if 'Packet Tracer' in col and 'Points Grade' in col:
            # Extract the activity name (before "Points Grade")
            activity_name = col.split(' Points Grade')[0]
            pt_columns.append((activity_name, col))
    return pt_columns

def match_students(brightspace_df, grades_df):
    """Match students between the two dataframes"""
    matches = []
    
    for idx, grade_row in grades_df.iterrows():
        # Try different approaches to find the filename
        student_name = None
        
        # Method 1: Try common column names
        possible_names = ['Filename', 'filename', 'File', 'file', 'File Name', 'filename ', ' Filename']
        for col_name in possible_names:
            if col_name in grades_df.columns:
                student_name = extract_student_name_from_filename(grade_row[col_name])
                if student_name:
                    break
        
        # Method 2: If that failed, try checking all columns for .pka files
        if not student_name:
            for col in grades_df.columns:
                cell_value = str(grade_row[col])
                if '.pka' in cell_value and len(cell_value) > 20:  # Likely a filename
                    student_name = extract_student_name_from_filename(cell_value)
                    if student_name:
                        break
        
        # Method 3: Try specific column index (third column is index 2)
        if not student_name and len(grades_df.columns) >= 3:
            student_name = extract_student_name_from_filename(grade_row.iloc[2])
        if student_name is None:
            continue
            
        # Try to match by name parts
        name_parts = student_name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            
            # Look for matching student in brightspace
            for bs_idx, bs_row in brightspace_df.iterrows():
                if (pd.notna(bs_row['First Name']) and pd.notna(bs_row['Last Name']) and
                    first_name.lower() in bs_row['First Name'].lower() and 
                    last_name.lower() in bs_row['Last Name'].lower()):
                    matches.append({
                        'brightspace_idx': bs_idx,
                        'grades_idx': idx,
                        'student_name': student_name,
                        'brightspace_name': f"{bs_row['First Name']} {bs_row['Last Name']}",
                        'grade': grade_row.get('Total Percentage', grade_row.get('Total_Percentage', grade_row.get('total percentage', None)))
                    })
                    break
    
    return matches

def match_students_transformed(brightspace_df, transformed_grades):
    """Match students using the transformed grades data"""
    matches = []
    
    for idx, grade_row in transformed_grades.iterrows():
        first_name = grade_row['First_Name']
        last_name = grade_row['Last_Name']
        grade = grade_row['Grade_Numeric']
        
        # Look for matching student in brightspace
        for bs_idx, bs_row in brightspace_df.iterrows():
            if (pd.notna(bs_row['First Name']) and pd.notna(bs_row['Last Name'])):
                # Try exact match first
                if (first_name.lower() == bs_row['First Name'].lower() and
                    last_name.lower() == bs_row['Last Name'].lower()):
                    matches.append({
                        'brightspace_idx': bs_idx,
                        'grades_idx': grade_row['Original_Index'],
                        'student_name': grade_row['Full_Name'],
                        'brightspace_name': f"{bs_row['First Name']} {bs_row['Last Name']}",
                        'grade': grade
                    })
                    break
                # Try partial match if exact fails
                elif (first_name.lower() in bs_row['First Name'].lower() and
                      last_name.lower() in bs_row['Last Name'].lower()):
                    matches.append({
                        'brightspace_idx': bs_idx,
                        'grades_idx': grade_row['Original_Index'],
                        'student_name': grade_row['Full_Name'],
                        'brightspace_name': f"{bs_row['First Name']} {bs_row['Last Name']}",
                        'grade': grade
                    })
                    break
    
    return matches

def main():
    st.title("Brightspace Packet Tracer Grade Processor")
    st.write("Upload your Brightspace CSV and Grade Details CSV to process Packet Tracer grades.")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Brightspace CSV")
        brightspace_file = st.file_uploader(
            "Upload Brightspace CSV", 
            type=['csv'],
            key="brightspace"
        )
    
    with col2:
        st.subheader("Grade Details CSV")
        grades_file = st.file_uploader(
            "Upload Grade Details CSV", 
            type=['csv'],
            key="grades"
        )
    
    if brightspace_file is not None and grades_file is not None:
        try:
            # Load the CSV files
            brightspace_df = pd.read_csv(brightspace_file)
            grades_df = pd.read_csv(grades_file)
            
            # Clean column names in grades_df to remove extra spaces
            grades_df.columns = grades_df.columns.str.strip()
            
            # Transform the grades CSV for better matching
            transformed_grades = transform_grades_csv(grades_df)
            
            st.success("Files uploaded successfully!")
            
            # Show transformation results
            if len(transformed_grades) > 0:
                st.subheader("📋 Transformed Grade Details")
                st.write(f"Successfully processed {len(transformed_grades)} student submissions:")
                
                # Show the transformed data
                display_df = transformed_grades[['Full_Name', 'First_Name', 'Last_Name', 'Grade']].copy()
                st.dataframe(display_df)
                
                # Allow manual corrections
                st.subheader("✏️ Manual Name Corrections (Optional)")
                st.write("If any names don't match exactly with Brightspace, you can make corrections:")
                
                corrections = {}
                for idx, row in transformed_grades.iterrows():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Original:** {row['Full_Name']}")
                    with col2:
                        corrected_first = st.text_input(
                            f"First Name",
                            value=row['First_Name'],
                            key=f"first_{idx}"
                        )
                    with col3:
                        corrected_last = st.text_input(
                            f"Last Name",
                            value=row['Last_Name'],
                            key=f"last_{idx}"
                        )
                    
                    if corrected_first != row['First_Name'] or corrected_last != row['Last_Name']:
                        corrections[idx] = {
                            'first_name': corrected_first,
                            'last_name': corrected_last
                        }
                
                # Apply corrections
                if corrections:
                    for idx, correction in corrections.items():
                        transformed_grades.loc[idx, 'First_Name'] = correction['first_name']
                        transformed_grades.loc[idx, 'Last_Name'] = correction['last_name']
                        transformed_grades.loc[idx, 'Full_Name'] = f"{correction['first_name']} {correction['last_name']}"
            else:
                st.error("Could not extract student names from the Grade Details file. Please check the file format.")
                st.write("**Raw CSV Data for debugging:**")
                st.dataframe(grades_df)
                st.stop()
            
            # Display basic info about the files
            st.subheader("File Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Brightspace CSV:** {len(brightspace_df)} students")
                st.write(f"Columns: {len(brightspace_df.columns)}")
            
            with col2:
                st.write(f"**Grade Details CSV:** {len(grades_df)} submissions")
                st.write(f"Columns: {len(grades_df.columns)}")
                
            # Debug: Show column names and sample data
            with st.expander("Debug: Column Names and Data"):
                st.write("**Grades CSV Columns:**")
                for i, col in enumerate(grades_df.columns):
                    st.write(f"{i}: '{col}'")
                
                st.write("**Sample Raw Data:**")
                st.dataframe(grades_df.head(3))
                
                st.write("**Filename Column Analysis:**")
                filename_cols = [col for col in grades_df.columns if 'filename' in col.lower() or 'file' in col.lower()]
                st.write(f"Potential filename columns: {filename_cols}")
                
                # Try to show data from all columns to see where the full filename might be
                st.write("**All Column Data (first row):**")
                for col in grades_df.columns:
                    sample_val = str(grades_df[col].iloc[0]) if len(grades_df) > 0 else "No data"
                    st.write(f"  '{col}': '{sample_val}'")
                
                # Try the third column which typically contains filename in your sample
                if len(grades_df.columns) >= 3:
                    col_name = grades_df.columns[2]  # Index 2 (third column)
                    st.write(f"**Testing column index 2 ('{col_name}'):**")
                    for i, val in enumerate(grades_df[col_name].head(5)):
                        extracted = extract_student_name_from_filename(val)
                        st.write(f"  {i}: '{val}' → '{extracted}'")
            
            # Extract Packet Tracer activities
            pt_activities = extract_packet_tracer_columns(brightspace_df)
            
            if not pt_activities:
                st.error("No Packet Tracer activities found in the Brightspace CSV!")
                return
            
            st.subheader("Select Packet Tracer Activity")
            
            # Create a selectbox for activities
            activity_names = [name for name, col in pt_activities]
            selected_activity = st.selectbox(
                "Choose the Packet Tracer activity to update:",
                activity_names
            )
            
            # Get the corresponding column name
            selected_column = None
            for name, col in pt_activities:
                if name == selected_activity:
                    selected_column = col
                    break
            
            if selected_column:
                st.write(f"Selected activity: **{selected_activity}**")
                
                # Match students using transformed data
                matches = match_students_transformed(brightspace_df, transformed_grades)
                
                st.subheader("Student Matching Results")
                if matches:
                    st.write(f"Found {len(matches)} matching students:")
                    
                    # Display matches in a table
                    match_data = []
                    for match in matches:
                        match_data.append({
                            'Brightspace Name': match['brightspace_name'],
                            'Grade Details Name': match['student_name'],
                            'Grade': f"{match['grade']:.1f}%" if pd.notna(match['grade']) else 'No Grade'
                        })
                    
                    st.dataframe(pd.DataFrame(match_data))
                    
                    # Update grades section
                    st.subheader("Update Grades")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Update Grades", type="primary"):
                            # Create a copy of the brightspace dataframe
                            updated_df = brightspace_df.copy()
                            
                            # Update matched students
                            updates_made = 0
                            for match in matches:
                                if pd.notna(match['grade']):
                                    updated_df.loc[match['brightspace_idx'], selected_column] = match['grade']
                                    updates_made += 1
                            
                            st.success(f"Updated {updates_made} student grades!")
                            
                            # Store updated dataframe in session state
                            st.session_state['updated_df'] = updated_df
                            st.session_state['selected_column'] = selected_column
                    
                    with col2:
                        if st.button("Auto-load Zeros for Missing Grades"):
                            if 'updated_df' in st.session_state:
                                updated_df = st.session_state['updated_df'].copy()
                            else:
                                updated_df = brightspace_df.copy()
                            
                            # Find students with missing grades (empty or NaN)
                            missing_mask = (updated_df[selected_column].isna() | 
                                          (updated_df[selected_column] == '') | 
                                          (updated_df[selected_column] == ' '))
                            
                            zeros_added = missing_mask.sum()
                            updated_df.loc[missing_mask, selected_column] = 0
                            
                            st.success(f"Added zeros for {zeros_added} missing grades!")
                            
                            # Store updated dataframe in session state
                            st.session_state['updated_df'] = updated_df
                            st.session_state['selected_column'] = selected_column
                    
                    # Download section
                    if 'updated_df' in st.session_state:
                        st.subheader("Download Updated CSV")
                        
                        # Show preview of changes
                        st.write("**Preview of updated grades:**")
                        updated_df = st.session_state['updated_df']
                        selected_column = st.session_state['selected_column']
                        
                        # Show only students with non-empty values in the selected column
                        preview_df = updated_df[['Last Name', 'First Name', selected_column]].copy()
                        preview_df = preview_df[preview_df[selected_column].notna() & 
                                               (preview_df[selected_column] != '') & 
                                               (preview_df[selected_column] != ' ')]
                        
                        st.dataframe(preview_df)
                        
                        # Create download button
                        csv_buffer = StringIO()
                        updated_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="Download Updated Brightspace CSV",
                            data=csv_data,
                            file_name="updated_brightspace.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning("No matching students found between the two files.")
                    st.write("**Possible issues:**")
                    st.write("- Student names in the grade details filename don't match Brightspace names")
                    st.write("- Check if the filename format in grade details is correct")
                    
                    # Show sample data for debugging
                    with st.expander("Debug Information"):
                        if len(transformed_grades) > 0:
                            st.write("**Transformed Grade Details:**")
                            st.dataframe(transformed_grades[['Full_Name', 'First_Name', 'Last_Name', 'Grade']])
                        
                        st.write("**Sample Brightspace Student Names:**")
                        sample_students = brightspace_df[['First Name', 'Last Name']].head(5)
                        st.dataframe(sample_students)
                        
                        st.write("**Name Matching Attempts:**")
                        for idx, grade_row in transformed_grades.iterrows():
                            st.write(f"Trying to match: {grade_row['First_Name']} {grade_row['Last_Name']}")
                            found_match = False
                            for bs_idx, bs_row in brightspace_df.iterrows():
                                if (pd.notna(bs_row['First Name']) and pd.notna(bs_row['Last Name'])):
                                    if (grade_row['First_Name'].lower() == bs_row['First Name'].lower() and
                                        grade_row['Last_Name'].lower() == bs_row['Last Name'].lower()):
                                        st.write(f"  ✅ Exact match: {bs_row['First Name']} {bs_row['Last Name']}")
                                        found_match = True
                                        break
                                    elif (grade_row['First_Name'].lower() in bs_row['First Name'].lower() and
                                          grade_row['Last_Name'].lower() in bs_row['Last Name'].lower()):
                                        st.write(f"  ⚠️ Partial match: {bs_row['First Name']} {bs_row['Last Name']}")
                                        found_match = True
                                        break
                            if not found_match:
                                st.write(f"  ❌ No match found")
                            if idx >= 2:  # Limit to first 3 for brevity
                                break
        
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.write("Please check that your CSV files are properly formatted.")

if __name__ == "__main__":
    main()