import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import urllib.parse  # URL 인코딩
from sklearn.metrics.pairwise import cosine_similarity

# CSV 파일 로드 (데이터셋)
file_path = "file/path"
data = pd.read_csv(file_path, encoding='cp949')

#  Word2Vec 벡터 데이터 로드
word2vec_path = r"C:/Users/minjw/2025/khuda/toyproj/word2vec_vectors.xlsx"  # 실제 경로 입력
word_vectors_df = pd.read_excel(word2vec_path, index_col=0)

vector_size = word_vectors_df.shape[1]  # Word2Vec 벡터 차원 확인

st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 1])

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 키워드 기반 음식점 필터링 함수
def filter_and_rank_stores(keyword, data, top_n=3):
    filtered_df = data[data['menu_listed'].str.contains(keyword, na=False, case=False)]
    
    if len(filtered_df) >= 20:
        # 20개 이상이면 기존처럼 상위 3개만 반환
        top_stores = filtered_df.sort_values(by='y_pred', ascending=False).head(top_n)
        return top_stores[['store_name', 'address', 'menu_listed', 'y_pred', 'distance', 'average_price']], None
    else:
        # 20개 미만이면 유사 메뉴 추천 필요
        return None, recommend_similar_menus(keyword)

# 유사한 메뉴 추천 함수 (Word2Vec 기반 + 가게 연결)
def recommend_similar_menus(input_word, top_n=20, similarity_threshold=0.5):
    matched_menus = [word for word in word_vectors_df.index if input_word in str(word)]
    matched_menus_with_stores = []

    # 입력된 단어를 포함하는 메뉴가 있는 가게 찾기
    for menu in matched_menus:
        matched_stores = data[data['menu_listed'].str.contains(menu, na=False, case=False)]['store_name'].unique()
        for store in matched_stores:
            matched_menus_with_stores.append(f"{store} - {menu}")

    # 20개 미만이면, 추가적으로 연관된 메뉴 검색 (어절 확장)
    if len(matched_menus_with_stores) < top_n:
        additional_menus = [
            str(word) for word in word_vectors_df.index
            if isinstance(word, str) and input_word[:-1] in word and word not in matched_menus
        ]

        for menu in additional_menus:
            matched_stores = data[data['menu_listed'].str.contains(menu, na=False, case=False)]['store_name'].unique()
            for store in matched_stores:
                matched_menus_with_stores.append(f"{store} - {menu}")

    # 20개 미만이면, Word2Vec 기반 유사 메뉴 추가 추천
    if len(matched_menus_with_stores) < top_n:
        try:
            input_vector = word_vectors_df.loc[input_word].values.reshape(1, -1)
            similarity_scores = cosine_similarity(input_vector, word_vectors_df.values)[0]
            similar_menus_sorted = sorted(
                zip(word_vectors_df.index, similarity_scores),
                key=lambda x: x[1], reverse=True
            )

            # 유사도가 일정 기준 이상인 단어만 추천
            similar_menus = [menu for menu, score in similar_menus_sorted if score > similarity_threshold and menu not in matched_menus]

            for menu in similar_menus:
                matched_stores = data[data['menu_listed'].str.contains(menu, na=False, case=False)]['store_name'].unique()
                for store in matched_stores:
                    matched_menus_with_stores.append(f"{store} - {menu}")

        except KeyError:
            pass  # 입력 키워드가 Word2Vec 벡터에 없는 경우 대비

    # 최종 추천 리스트 (최대 top_n개) - 중복 제거 및 정렬
    matched_menus_with_stores = list(set(matched_menus_with_stores))  # 중복 제거
    return matched_menus_with_stores[:top_n]


# 세션 상태 초기화 (선택한 가게 저장용)
if "selected_store" not in st.session_state:
    st.session_state["selected_store"] = "경희대학교 국제캠퍼스"  # 기본값

if "selected_address" not in st.session_state:
    st.session_state["selected_address"] = ""  # 기본 주소 초기화

# 카카오맵 업데이트 함수 (주소 기반 검색)
def update_map(store_name, store_address):
    st.session_state["selected_store"] = store_name
    st.session_state["selected_address"] = store_address  # 선택한 가게의 주소 저장
    st.rerun()  # 화면 즉시 갱신하여 반영

with col1:
    # 현재 선택된 가게의 "주소"를 기반으로 카카오맵 검색 URL 생성
    search_query = st.session_state.get("selected_address", "") if st.session_state.get("selected_address", "") else st.session_state.get("selected_store", "경희대학교 국제캠퍼스")
    kakao_map_url = f"https://map.kakao.com/?q={urllib.parse.quote(search_query)}"

    # 카카오맵 iframe 삽입 (업데이트된 지도 반영)
    st.markdown(
        f"""
        <div style="width: 100%; display: flex; justify-content: center;">
            <iframe src="{kakao_map_url}" width="100%" height="500" style="border:none;"></iframe>
        </div>
        """,
        unsafe_allow_html=True
    )
    store_data = st.session_state.get("store_data", {})
    selected_store = st.session_state.get("selected_store", None)  # 세션 상태에서 가져오기
    
    if selected_store and selected_store in store_data:
        store_info = store_data[selected_store]
        
        st.subheader(f"📍 {selected_store} 정보")
        st.write(f"**주소**: {store_info['address']}")
        st.write(f"**평균 가격**: {store_info['price']} 원")
        st.write(f"**거리**: {store_info['distance']} km")

with col2:
    st.title('🍽 음식점 필터링 시스템')
    keyword = st.text_input("🔍 찾고 싶은 메뉴를 입력하세요:")
    if keyword:
        top_stores, similar_menus = filter_and_rank_stores(keyword, data)

        if top_stores is not None:
            st.write(f"🔍 '{keyword}'을 포함하는 음식점 목록 (상위 3개):")
            st.dataframe(top_stores)

            # 음식점 데이터 저장
            store_data = {}
            for i, row in top_stores.iterrows():
                store_name = row["store_name"]
                store_address = row["address"]  # 주소 추가
                store_data[store_name] = {
                    "address": store_address,
                    "distance": float(row["distance"]),
                    "price": float(row["average_price"])
                }
            
            st.session_state["store_data"] = store_data

            # 유사 가게 평균 계산
            similar_store_avg = {
                "distance": top_stores["distance"].mean(),
                "price": top_stores["average_price"].mean()
            }

            # 선택한 가게를 클릭하면 지도 업데이트 (주소 기반 검색)
            for i, store in enumerate(top_stores["store_name"]):
                store_address = store_data[store]["address"]  # 주소 가져오기
                if st.button(f"🥇 {i+1}위: {store}", key=f"btn_{i}"):
                    update_map(store, store_address)  # 클릭하면 주소 기반으로 지도 검색

            # 선택된 가게가 있으면 비교 그래프 생성
            selected_store = st.session_state["selected_store"]
            # 🔹 가게 정보와 메뉴를 버튼 클릭 후 보여주기
            selected_store = st.session_state.get("selected_store", None)

            if selected_store in store_data:
                st.subheader(f"📊 {selected_store}의 비교 그래프")

                # 선택한 가게의 데이터 가져오기
                chosen_distance = store_data.get(selected_store, {}).get("distance", 0)
                chosen_price = store_data.get(selected_store, {}).get("price", 0)

                # 그래프 생성
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                # 거리 비교 그래프
                bars1 = ax[0].bar(["유사 가게 평균", selected_store], [similar_store_avg["distance"], chosen_distance], color=['gray', 'red'])
                ax[0].set_title("거리 비교 (km)")
                ax[0].bar_label(bars1, fmt="%.2f km", padding=3)

                # 가격 비교 그래프
                bars2 = ax[1].bar(["유사 가게 평균", selected_store], [similar_store_avg["price"], chosen_price], color=['gray', 'red'])
                ax[1].set_title("평균 가격 비교 (원)")
                ax[1].bar_label(bars2, fmt="%d 원", padding=3)

                st.pyplot(fig)

        elif similar_menus:
            st.write(f"🔎 '{keyword}'을 포함하는 음식점이 20개 미만입니다. 대신 유사한 메뉴를 추천합니다:")
            for menu in similar_menus:
                st.write(f"🍽 {menu}")

        else:
            st.write("해당 키워드를 포함하는 음식점이나 유사한 메뉴가 없습니다.")
