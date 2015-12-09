__author__ = 'ENG.AHMED HANI'
import csv


def set_data(res):
    output_file = open("./YWD_res.csv", "wb")
    writer = csv.writer(output_file)
    writer.writerow(
        ["Id",
         'ARSON',
         'ASSAULT',
         'BAD CHECKS',
         'BRIBERY',
         'BURGLARY',
         'DISORDERLY CONDUCT',
        'DRIVING UNDER THE INFLUENCE',
         'DRUG/NARCOTIC',
         'DRUNKENNESS',
         'EMBEZZLEMENT',
         'EXTORTION',
        'FAMILY OFFENSES',
         'FORGERY/COUNTERFEITING',
         'FRAUD',
         'GAMBLING',
         'KIDNAPPING',
         'LARCENY/THEFT',
        'LIQUOR LAWS','LOITERING',
         'MISSING PERSON',
         'NON-CRIMINAL',
         'OTHER OFFENSES',
        'PORNOGRAPHY/OBSCENE MAT',
         'PROSTITUTION',
         'RECOVERED VEHICLE',
         'ROBBERY','RUNAWAY',
        'SECONDARY CODES',
         'SEX OFFENSES FORCIBLE',
         'SEX OFFENSES NON FORCIBLE',
         'STOLEN PROPERTY',
        'SUICIDE',
         'SUSPICIOUS OCC',
         'TREA',
         'TRESPASS',
         'VANDALISM',
         'VEHICLE THEFT',
         'WARRANTS',
        'WEAPON LAWS'])

    for x in range(len(res)):
        res[x].insert(0, x)
        writer.writerow(round(res[x], 3))

    output_file.close()


