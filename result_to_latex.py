import pandas as pd

if __name__ == '__main__':
	ride_pdf = pd.read_csv('results/ride.csv')
	ride_pdf = ride_pdf.loc[:, ride_pdf.columns!='Unnamed: 0']
	ride_latex_code = ride_pdf.to_latex(index=False,float_format="%.3f")
	print(ride_latex_code)

	mm_pdf = pd.read_csv('results/mm.csv')
	mm_pdf = mm_pdf.loc[:, mm_pdf.columns!='Unnamed: 0']
	mm_latex_code = mm_pdf.to_latex(index=False,float_format="%.3f")
	print(mm_latex_code)

	kidney_pdf = pd.read_csv('results/kidney.csv')
	kidney_pdf = kidney_pdf.loc[:, kidney_pdf.columns!='Unnamed: 0']
	kidney_latex_code = kidney_pdf.to_latex(index=False,float_format="%.3f")
	print(kidney_latex_code)

