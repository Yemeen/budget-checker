import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from io import BytesIO


def find_text_pdf(search_string: str, pdf_text: str):
    match = re.search(fr"{re.escape(search_string)}(.*)", pdf_text)

    if match:
        text = match.group(1)
        return text.strip()
    else:
        return "Phrase not found."


def load_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = pdf.pages
        pdf_text = '\n'.join([p.extract_text() for p in pages])

    pdf_lines = pdf_text.split('\n')
    pdf_data = [line.split(',') for line in pdf_lines]

    return pdf_text, pdf_data


def monthly_services(pdf_text: str):
    match = re.search('Monthly Services\n(.*)Total Plan Cost',
                      pdf_text, re.DOTALL)

    if match is not None:
        result = match.group(1)
        # Split into lines
        lines = result.strip().split('\n')

        # Initialize a list to store rows
        data = []

        # Iterate over lines
        for i, line in enumerate(lines):
            split_line = line.split()

            # If line has more than 15 words:
            # last element starts with '$'
            if len(split_line) > 15 and split_line[-1].startswith('$'):
                service_cost = split_line[-1]  # The last element is cost
                # Elements before the 14 month numbers and cost form 'Service and Provider'
                provider = ' '.join(split_line[:-15])
                # The 14 month numbers and the additional column (excludes 'Service Cost')
                months_and_unit = split_line[-15:-1]
                row = [provider] + months_and_unit + [service_cost]
                data.append(row)
            else:
                # Diagnostic print statements
                print(f"Line {i+1} skipped. Check formatting.")

        # Define columns
        columns = ['Service and Provider'] + \
            ['M'+str(i+1) for i in range(13)] + ['Total Cost', 'Service Cost']

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    else:
        print("Monthly services not found!")


def app():
    st.title('Budget Data Explorer')

    # upload excel file
    excel_file = st.file_uploader("Upload excel file", type=['xlsx'])

    # upload pdf file
    pdf_file = st.file_uploader("Upload pdf file", type=['pdf'])

    if excel_file is not None and pdf_file is not None:
        excel_df = pd.read_excel(excel_file, header=None)
        pdf_text, pdf_data = load_pdf(BytesIO(pdf_file.read()))

        if excel_df is not None and pdf_text is not None:
            col1, col2 = st.columns(2)  # create two columns
            with col1:
                # extract and print properties from excel file
                st.subheader('Excel Properties')
                excel_properties = {
                    'dda_budget': excel_df.iloc[0, 9],
                    'pcp_status': excel_df.iloc[2, 8],
                    'waiver_type': excel_df.iloc[8, 10],
                    'effective_date': excel_df.iloc[4, 6],
                    'annual_plan_date': excel_df.iloc[4, 10],
                    'fmcs_agency': excel_df.iloc[12, 3],
                    'rate_month': excel_df.iloc[12, 6],
                    'num_months': excel_df.iloc[12, 8],
                    'alt_rate_month': excel_df.iloc[13, 6],
                    'alt_num_months': excel_df.iloc[13, 8],
                }

                for name, value in excel_properties.items():
                    st.write(f'**{name}**: {value}')

            with col2:
                # show properties
                st.subheader('PDF Properties')
                attrs = ['Annual Waiver Plan Services Total:',
                         'Recalculated:', 'Plan Type:', 'Program Type:', 'Annual PCP Date']
                for attr in attrs:
                    st.write(f'**{attr}** {find_text_pdf(attr, pdf_text)}')
                # show monthly services
                st.subheader('Monthly Services')
                st.dataframe(monthly_services(pdf_text), hide_index=True)
            st.subheader('Excel Data')
            services = pd.DataFrame(columns=excel_df.columns)
            for index, row in excel_df.iloc[17:].iterrows():
                third = row.iloc[3]
                if np.isreal(third) and np.isfinite(third):
                    services = pd.concat(
                        [services, pd.DataFrame([row])], ignore_index=True, axis=0)
            services.drop(columns=[0, 6, 7, 8, 10, 11], axis=1, inplace=True)
            st.dataframe(services)
            st.markdown('#### IDFGS')
            st.dataframe(excel_df.iloc[136:142, :])


if __name__ == '__main__':
    app()
