from flask import Flask, render_template, request
import pandas as pd
import pickle as pkl

data_set = pd.read_csv('model/cleaned_data.csv')

# Load the machine learning model from the Pickle file
with open('model/DTR_Model.pkl', 'rb') as file:
    model = pkl.load(file)


laptop_brand = data_set['brand'].unique()
processor_brand = data_set['processor_brand'].unique()
processor_name = data_set['processor_name'].unique()
processor_gnrtn = data_set['processor_gnrtn'].unique()
ram_gb = data_set['ram_gb'].unique()
ram_type = data_set['ram_type'].unique()
os = data_set['os'].unique()
hdd = data_set['hdd'].unique()
ssd = data_set['ssd'].unique()
graphic_card_gb = data_set['graphic_card_gb'].unique()
Touchscreen = data_set['Touchscreen'].unique()
msoffice = data_set['msoffice'].unique()
rating = data_set['rating'].unique()

data_set.info()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted = None
    pred_value=None
    
    brand=""
    Processor_Name=""
    Processor_Brand=""
    ms=""
    touchscreen=""
    Hdd=""
    Ssd=""
    Ram_gb=""
    Oss=""
    gc=""

    context=""
    if request.method == 'POST':
        # Retrieve form data
        brand = request.form.get('brand')
        Processor_Brand = request.form.get('pbrand')
        Processor_Name = request.form.get('pn')
        Processor_Generation = request.form.get('pg')
        Ram_type = request.form.get('ram_type')
        Ram_gb = request.form.get('ram_gb')
        Oss = request.form.get('os')
        Hdd = request.form.get('hdd')
        Ssd = request.form.get('ssd')
        gc = request.form.get('gcg')
        touchscreen = request.form.get('ts')
        ms = request.form.get('ms')
        Rating = request.form.get('rating')

        ##converting
        Ram_gb_int = int(Ram_gb)
        Processor_Generation_int = int(Processor_Generation)
        Hdd_int = int(Hdd)
        Ssd_int = int(Ssd)
        Rating_int = int(Rating)
        gc_int=int(gc)

        data = pd.DataFrame([[brand,Processor_Brand,Processor_Name,Processor_Generation_int,Ram_gb_int,Ram_type,Hdd_int,Ssd_int,Oss,gc_int,touchscreen,ms,Rating_int]],columns=['brand','processor_brand','processor_name','processor_gnrtn','ram_gb','ram_type','ssd','hdd','os','graphic_card_gb','Touchscreen','msoffice','rating'])

        ##linear model prediction on sample data
        predicted=model.predict(data)
    if predicted is not None:
        pred_value=float(predicted)    

    context = {
        'laptop_brand': laptop_brand,
        'pb': processor_brand,
        'pn': processor_name,
        'pg': processor_gnrtn,
        'ram_type': ram_type,
        'ram_gb': ram_gb,
        'os': os,
        'hdd': hdd,
        'ssd': ssd,
        'gcg': graphic_card_gb,
        'ts': Touchscreen,
        'ms': msoffice,
        'rating': rating,
        'predicted':predicted,
        'pred_value':pred_value,
        'lbs':brand,
        'pns':Processor_Name,
        'pbs':Processor_Brand,
        'mss':ms,
        'tss':touchscreen,
        'hdds':Hdd,
        'ssds':Ssd,
        'Rams':Ram_gb,
        'Oss':Oss,
        'gcgs':gc,
    }
    
    return render_template('index.html', context=context)
    


@app.route('/sample-predicted-laptop')
def sample_laptop_Pred():
    l_link=['https://imgeng.jagran.com/images/2023/mar/ASUS%20Vivobook%2016X%20(2022)1680007341926.jpg','https://www.lenovo.com/medias/lenovo-laptops-lenovo-v15-gen-3-15-intel-hero.png?context=bWFzdGVyfHJvb3R8NjIyNjU0fGltYWdlL3BuZ3xoMjEvaGZkLzE0NjYzMTk4NzM2NDE0LnBuZ3xmYjZiMjhmNzdkZDVkZmVmNWI2ZDc5YjFkNWI5ZGUwM2VjMGRiZTg4NDQ0YzE2NDlkZDkxNzU2ZGFmY2Y4ODAx','https://5.imimg.com/data5/SELLER/Default/2021/12/PH/NA/RW/8118327/acer-aspire-3-ryzen-3-a315-23-notebook-500x500.jpg','https://rukminim2.flixcart.com/image/850/1000/kg2l47k0/computer/r/s/w/avita-original-imafwdc3fftgchpu.jpeg?q=90','https://in-media.apjonlinecdn.com/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/8/1/81B47PA-6_T1683626142.png','https://m.media-amazon.com/images/I/81z8GkrHtNL._AC_UF894,1000_QL80_.jpg','https://img.etimg.com/photo/msid-96403101/asus-rog-strix-g15-laptop-.jpg','https://www.gravis.de/medias/sys_master/images/images/h42/h69/10812603826206/144126-1-product-3x.jpg']
    
    laptop_price=data_set['Price'].sort_values(ascending=True).head(8)
    context = {
        'laptops': list(zip(laptop_brand, l_link, laptop_price))
    }
    return render_template('predicted.html',context=context)

if __name__ == '__main__':
    app.run(debug=True)
