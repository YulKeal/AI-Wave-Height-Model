import cdsapi

def download_data(year):
    c = cdsapi.Client()
    filename = 'F:/xxx/' + str(year) + '.nc'
    print(f"Downloading data for {year} into {filename}")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'area': [
                70, -180, -70,
                180,
            ],
            'grid': [0.5, 0.5],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'year': str(year),
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 'significant_height_of_combined_wind_waves_and_swell',
            ],
        }, filename)
    print(f"Finished downloading data for {year}")

def main():
    download_data(2020)

if __name__ == "__main__":
    main()
