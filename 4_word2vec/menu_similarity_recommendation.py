import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 저장된 Word2Vec 벡터 불러오기 Important: 파일 로컬 경로 입력하기!!
file_path = r"file/path"  # 여기에 실제 파일 경로 입력. Important: 반드시 앞에 r 붙일 것!
word_vectors_df = pd.read_excel(file_path, index_col=0)

# Word2Vec 벡터 차원 확인
vector_size = word_vectors_df.shape[1]  # 벡터 차원 고정

# IMPORTANT: 새로운 키워드 입력
input_word = "돼지"  # 예제 입력

# 입력 단어를 포함하는 메뉴 찾기
matched_menus = [str(word) for word in word_vectors_df.index if isinstance(word, str) and input_word in word]

# 추천된 메뉴가 20개 미만이면 추가 확장 검색 (연관된 단어 포함)
additional_menus = []
if len(matched_menus) < 20:
    print("\n추천된 메뉴가 부족하여 추가 검색을 진행합니다...")
    extended_matched_menus = [
        str(word) for word in word_vectors_df.index if isinstance(word, str) and input_word[:-1] in word
    ]
    additional_menus = list(set(extended_matched_menus) - set(matched_menus))

# 유사도 계산 및 추천
print(f"\n'{input_word}'과(와) 유사한 메뉴 추천:")

# `input_word`가 포함된 메뉴는 반드시 상위 고정
matched_vectors = np.array([
    word_vectors_df.loc[menu].values for menu in matched_menus if menu in word_vectors_df.index
    and word_vectors_df.loc[menu].shape[0] == vector_size  # 벡터 크기 체크
])

# 벡터가 존재하지 않는 경우 대비 (차원 불일치 방지)
if matched_vectors.shape[0] > 0:
    avg_matched_vector = np.mean(matched_vectors, axis=0).reshape(1, vector_size)
    similarity_scores = cosine_similarity(avg_matched_vector, matched_vectors)[0]

    matched_menus_with_scores = sorted(zip(matched_menus, similarity_scores), key=lambda x: x[1], reverse=True)
    
    print("\n 입력 단어 포함 메뉴 추천")
    for menu, score in matched_menus_with_scores:
        print(f"{menu}: {score:.4f}")

else:
    print("\n 입력 단어 포함 메뉴가 없습니다.")

# 추가 검색된 메뉴로 20개까지 채우기
if len(additional_menus) > 0:
    remaining_slots = 20 - len(matched_menus)
    final_additional_menus = additional_menus[:remaining_slots]

    additional_vectors = np.array([
        word_vectors_df.loc[menu].values for menu in final_additional_menus if menu in word_vectors_df.index
        and word_vectors_df.loc[menu].shape[0] == vector_size  # 벡터 크기 체크
    ])

    if additional_vectors.shape[0] > 0:
        avg_additional_vector = np.mean(additional_vectors, axis=0).reshape(1, vector_size)
        additional_similarity_scores = cosine_similarity(avg_additional_vector, additional_vectors)[0]

        additional_menus_with_scores = sorted(zip(final_additional_menus, additional_similarity_scores), key=lambda x: x[1], reverse=True)

        print("\n 추가 검색된 연관 메뉴 추천")
        for menu, score in additional_menus_with_scores:
            print(f"{menu}: {score:.4f}")
    else:
        print("\n 추가 검색된 연관 메뉴가 없습니다.")

# 추가 추천이 부족할 경우, 벡터 유사도 기반 추가 추천
if len(matched_menus) + len(additional_menus) < 20:
    print("\n유사한 메뉴 추가 추천")

    similar_vectors = np.array([
        word_vectors_df.loc[word].values for word in word_vectors_df.index
        if isinstance(word, str) and any(char in word for char in input_word[:-1])
        and word_vectors_df.loc[word].shape[0] == vector_size  # 벡터 크기 체크
    ])

    if similar_vectors.shape[0] > 0:
        avg_similar_vector = np.mean(similar_vectors, axis=0).reshape(1, vector_size)
        similarity_scores = cosine_similarity(avg_similar_vector, word_vectors_df.values)[0]

        similar_menus_sorted = sorted(zip(word_vectors_df.index, similarity_scores), key=lambda x: x[1], reverse=True)[:10]

        print("\n 벡터 유사도 기반 추가 추천")
        for menu, score in similar_menus_sorted:
            print(f"{menu}: {score:.4f}")
    else:
        print("\n 벡터 유사도 기반 추가 추천 불가")

# 만약에 "관련된 단어를 찾을 수 없습니다."가 출력이 된다면, 단어 입력 받을 때 따옴표의 유무가 영향을 준 거라고 보시면 됩니다.
'''
그리고 메뉴를 전달할 때, 추천해주는 파일에서는 아마도 메뉴명에 따옴표가 해당되어있을텐데, 지금 이 코드에서는 일부러 따옴표를 제거한 상태라서
만약 오류가 뜨면 따옴표를 달고 있는 상태로 전달해주면 해결될 겁니다.
'''
