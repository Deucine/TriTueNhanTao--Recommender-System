import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
import os
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import streamlit as st

# Đọc dữ liệu từ các file CSV
credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')

# Hàm làm sạch tiêu đề
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# Hợp nhất hai DataFrame dựa trên tiêu đề
movies_df = movies_df.merge(credits_df, on='title')
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Loại bỏ các dòng chứa giá trị null
movies_df.dropna(inplace=True)

# Chuyển đổi giá trị trường thành 1 danh sách
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)

# Chuyển đổi giá trị trường thành 1 danh sách, giới hạn tối đa 3 phần tử
def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies_df['cast'] = movies_df['cast'].apply(convert_cast)

# Hàm lấy tên đạo diễn
def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return i['name']
    return ''

movies_df['director'] = movies_df['crew'].apply(fetch_director)

# Làm sạch và chuẩn hóa các cột
movies_df['overview'] = movies_df['overview'].apply(lambda x: str(x).split() if isinstance(x, str) else [])
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['director'] = movies_df['director'].apply(lambda x: x.replace(" ", ""))

# Tạo cột 'tags' bằng cách kết hợp các thể loại, diễn viên, đạo diễn và từ khóa
movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['director'].apply(lambda x: [x])

# Tạo DataFrame mới chỉ chứa các cột cần thiết
new_df = movies_df[['movie_id', 'title', 'tags']]

# Kết hợp các tags thành chuỗi và chuyển thành chữ thường
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# Hàm Stemmer
ps = PorterStemmer()
def stem(text):
    y = [ps.stem(i) for i in text.split()]
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

@st.cache_data
def vectorize_and_calculate_similarity(new_df):
    # Vector hóa các tags
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    # Tính toán ma trận độ tương đồng cosine
    similarity = cosine_similarity(vectors)
    return vectors, similarity

# Tính toán vectors và similarity trước khi lưu
vectors, similarity = vectorize_and_calculate_similarity(new_df)

# Tạo thư mục 'model' nếu chưa tồn tại
if not os.path.exists('model'):
    os.makedirs('model')

# Lưu mô hình sử dụng pickle
pickle.dump(movies_df, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

# Load mô hình đã lưu
movies = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# Hàm gọi API của TMDb để lấy thông tin phim
def fetch_movie_details(movie_title, api_key):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(search_url)
    data = response.json()
    if data['results']:
        movie_details = data['results'][0]
        return movie_details
    else:
        return None

# API key của TMDb
tmdb_api_key = 'f14e7a213888a71a4c0fc62acbf44370'  # Đã thay thế bằng API key của bạn

# Hàm gợi ý phim
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Khởi tạo danh sách phim yêu thích
if 'favorite_movies' not in st.session_state:
    st.session_state['favorite_movies'] = []

if 'watched_movies' not in st.session_state:
    st.session_state['watched_movies'] = []

# CSS tùy chỉnh
st.markdown(
    """
    <style>
    .custom-title {
        display: inline-flex; /* Sắp xếp các phần tử trên 1 hàng */
        align-items: center; /* Canh giữa theo chiều dọc */
        justify-content: center; /* Canh giữa theo chiều ngang */
        gap: 10px; /* Khoảng cách giữa các phần tử */
        font-family: 'Arial', sans-serif; /* Đổi font chữ */
        font-size: 48px; /* Kích thước chữ */
        font-weight: bold; /* Chữ đậm */
        color: red; /* Màu chữ đỏ */
        text-shadow: 2px 2px 4px #000000; /* Hiệu ứng bóng */
        margin-top: 20px;
        margin-bottom: 20px;
        white-space: nowrap; /* Đảm bảo không xuống dòng */
    }
    .custom-title img {
        height: 50px; /* Chiều cao của biểu tượng */
    }
    .stButton>button {
        background-color: #ff6347;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ff4500;
    }
    .stSelectbox>div {
        color: #ff6347;
        border-radius: 8px;
    }
    .css-1y0tads {
        background-color: #ffffff;
        border-radius: 8px;
    }
    .stMarkdown {
        color: red;
    }
    .cute-sticker {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .cute-sticker img {
        width: 80px;
        height: 80px;
        margin: 0 10px;
    }
    .white-text {
        color: white; /* Đổi màu chữ thành màu trắng */
    }
    .poster-container {
        position: relative;
        display: inline-block;
        width: 300px; /* Tăng kích thước poster */
        margin-right: 30px; /* Tăng khoảng cách giữa các poster */
        margin-bottom: 40px; /* Tăng khoảng cách dưới mỗi poster */
    }
    .info-box {
        position: absolute;
        top: 10px;
        left: 110%;
        background-color: rgb(0, 167, 153);
        color: white;
        padding: 10px;
        font-size: 14px;
        border-radius: 5px;
        display: none;
        z-index: 10;
        width: 250px;
        text-align: left;
    }
    .poster-container:hover .info-box {
        display: block;
    }
    .genre {
        font-size: 12px; /* Giảm kích thước chữ thể loại */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Giao diện Streamlit
st.markdown(
    '<div class="custom-title">🎬 <span>RECOMMENDATION SYSTEM</span> 🎥</div>', 
    unsafe_allow_html=True
)
st.markdown('<p class="white-text">Chọn một bộ phim bạn yêu thích và chúng tôi sẽ gợi ý các phim tương tự.</p>', unsafe_allow_html=True)

# Khởi tạo danh sách phim yêu thích
if 'favorite_movies' not in st.session_state:
    st.session_state.favorite_movies = []

# Khởi tạo biến lưu trữ trạng thái gợi ý và thông báo
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'notification' not in st.session_state:
    st.session_state.notification = None

# Hàm thêm phim vào danh sách yêu thích
def add_to_favorites(movie_title):
    if movie_title not in st.session_state.favorite_movies:
        st.session_state.favorite_movies.append(movie_title)
        st.session_state.notification = f"{movie_title} đã được thêm vào danh sách yêu thích!"
        st.session_state.show_recommendations = False
    else:
        st.session_state.notification = f"{movie_title} đã có trong danh sách yêu thích của bạn!"

# Danh sách phim và lựa chọn phim
movie_list = movies_df['title'].tolist()
selected_movie = st.selectbox('Chọn phim', movie_list, key='selected_movie')


col1, col2 = st.columns([3, 1])  # Tạo 2 cột với tỷ lệ 3:1 để nút YÊU THÍCH nằm bên phải

with col1:
    if st.button('GỢI Ý'):
        if selected_movie:
            # Get movie recommendations
            recommendations = recommend(selected_movie)
            
            if recommendations:
                st.write('## Các phim gợi ý:')
                st.markdown("<hr>", unsafe_allow_html=True)
                
                cols = st.columns(3)  # Create 5 columns to display movie recommendations horizontally
                
                for idx, (col, rec) in enumerate(zip(cols, recommendations)):
                    with col:
                        # Fetch movie details
                        movie_details = fetch_movie_details(rec, tmdb_api_key)
                        
                        if movie_details:
                            # Display movie poster and details
                            poster_url = f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
                            st.image(poster_url, use_container_width=True)
                            # Create clickable link for IMDb
                            imdb_url = f"https://www.imdb.com/find?q={movie_details['title'].replace(' ', '+')}"
                            st.markdown(f"[**{movie_details['title']}**]({imdb_url})", unsafe_allow_html=True)
                            st.markdown(f"Đánh giá: {round(movie_details['vote_average'], 1)}/10")
                            st.markdown(f"Ngày phát hành: {movie_details['release_date']}")
                            
                            # Properly formatted clickable link for the trailer
                            trailer_url = f"https://www.youtube.com/results?search_query={movie_details['title'].replace(' ', '+')}+trailer"
                            st.markdown(f"[Xem trailer]({trailer_url})")
                        else:
                            st.write(f"Không thể tìm thấy thông tin cho phim: {rec}.")
            else:
                st.write("Không có gợi ý phim nào. Vui lòng thử lại sau.")
        else:
            st.write('Vui lòng chọn một bộ phim để nhận gợi ý.')

with col2:
    if st.button('YÊU THÍCH'):
        if selected_movie:
            add_to_favorites(selected_movie)

 #Hiển thị danh sách yêu thích ở sidebar
with st.sidebar:
    st.write('## Phim yêu thích:')
    for favorite_movie in st.session_state.favorite_movies:
        st.markdown(f"- {favorite_movie}")

  

# Thêm chức năng lọc thể loại
genres = list(set([genre for sublist in movies_df['genres'].tolist() for genre in sublist]))
selected_genre = st.sidebar.selectbox('Chọn thể loại', genres)

# Lọc phim theo thể loại đã chọn
filtered_movies = movies_df[movies_df['genres'].apply(lambda x: selected_genre in x)]

# Hiển thị danh sách phim thuộc thể loại này trong giao diện chính
st.write(f"## Danh sách phim thuộc thể loại '{selected_genre}'")

columns = st.columns(3)  # Tạo 3 cột để hiển thị poster phim

for idx, title in enumerate(filtered_movies['title'].tolist()):
    movie_details = fetch_movie_details(title, tmdb_api_key)
    if movie_details:
        poster_url = f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
        movie_link = f"https://www.themoviedb.org/movie/{movie_details['id']}"
        with columns[idx % 3]:  # Hiển thị poster theo hàng, mỗi hàng có 3 poster
            st.markdown(
                f"""
                <div class="poster-container">
                    <a href="{movie_link}" target="_blank">
                        <img src="{poster_url}" style="width:70%">
                    </a>
                    <div class="info-box">
                        <p><strong>Năm sản xuất:</strong> {movie_details.get('release_date', 'N/A').split('-')[0]}</p>
                        <p><strong>Mức độ đánh giá:</strong> {round(movie_details.get('vote_average', 'N/A'))} /10</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write(f"Không có thông tin cho phim: {title}")





