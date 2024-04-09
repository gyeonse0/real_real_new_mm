from haversine import haversine
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

# 주어진 좌표값 (위도, 경도)
coordinates = np.array([
(	37.49267955	,	127.0306305	)	,
(	37.49233075	,	127.0745897	)	,
(	37.50528056	,	127.0289194	)	,
(	37.46682003	,	127.0969337	)	,
(	37.48005839	,	127.0641545	)	,
(	37.52269079	,	127.0328729	)	,
(	37.49828674	,	127.0301038	)	,
(	37.48811677	,	127.1030688	)	,
(	37.49012921	,	127.071565	)	,
(	37.48483867	,	127.0897795	)	,
(	37.46137037	,	127.1040205	)	,
(	37.50396606	,	127.0429099	)	,
(	37.4885025	,	127.0865469	)	,
(	37.49828846	,	127.0301032	)	,
(	37.51643246	,	127.0203269	)	,
(	37.47056578	,	127.1193125	)	,
(	37.52282078	,	127.0561406	)	,
(	37.52460119	,	127.0210076	)	,
(	37.48797908	,	127.1056314	)	,
(	37.49213604	,	127.0997527	)	,
(	37.5030427	,	127.0541314	)	,
(	37.5131779	,	127.065386	)	,
(	37.49812635	,	127.050691	)	,
(	37.49292753	,	127.0902766	)	,
(	37.49671719	,	127.071719	)	,
(	37.4893922	,	127.0506344	)	,
(	37.50796024	,	127.0303213	)	,
(	37.50288135	,	127.0413565	)	,
(	37.50015065	,	127.0334563	)	,
(	37.47463125	,	127.1161223	)	,
(	37.50937616	,	127.0423641	)	,
(	37.52794736	,	127.0438666	)	,
(	37.51909256	,	127.0496091	)	,
(	37.49835748	,	127.0614696	)	,
(	37.46450499	,	127.1026803	)	,
(	37.52012517	,	127.0447158	)	,
(	37.5103126	,	127.0463796	)	,
(	37.47974583	,	127.0878269	)	,
(	37.50463416	,	127.0250313	)	,
(	37.51871603	,	127.0469585	)	,
(	37.50159765	,	127.0377229	)	,
(	37.51642118	,	127.0304472	)	,
(	37.49086414	,	127.0553257	)	,
(	37.52643091	,	127.0284815	)	,
(	37.46287766	,	127.1014443	)	,
(	37.46438764	,	127.104165	)	,
(	37.48541153	,	127.0547066	)	,
(	37.49566238	,	127.0409102	)	,
(	37.52813664	,	127.0318383	)	,
(	37.5099099	,	127.0631778	)	,
])

# 맨해튼 거리를 저장할 행렬 초기화
haversine_manhattan_distances_matrix = np.zeros((50, 50))

# 좌표 간의 맨해튼 거리 계산 (가로 세로 각각에 대해 하버사인 적용)
for i in range(50):
    for j in range(50):
        if i != j:
            # 좌표 간의 가로, 세로 거리 계산
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            horizontal_distance = haversine((lat1, lon1), (lat1, lon2))
            vertical_distance = haversine((lat1, lon2), (lat2, lon2))
            
            # 가로 세로 거리의 합을 맨해튼 거리로 사용
            haversine_manhattan_distance = horizontal_distance + vertical_distance
            
            # 맨해튼 거리를 행렬에 저장
            haversine_manhattan_distances_matrix[i, j] = haversine_manhattan_distance

# 맨해튼 거리 데이터프레임 생성
df_manhattan = pd.DataFrame(haversine_manhattan_distances_matrix)

# 엑셀 파일로 저장 (바탕 화면 경로)
desktop_path = "C:/Users/User/OneDrive/바탕 화면/"
df_manhattan.to_excel(desktop_path + 'haversine_manhattan_distances.xlsx', index=False)

# 유클리드 거리를 저장할 행렬 초기화
haversine_distances_matrix = np.zeros((50, 50))

# 좌표 간의 유클리드 거리 계산 (haversine 함수 적용)
for i in range(50):
    for j in range(50):
        if i != j:
            # 주석으로 표시된 부분이 행렬의 값입니다.
            haversine_distances_matrix[i, j] = haversine(coordinates[i], coordinates[j])

# 유클리드 거리 데이터프레임 생성
df_euclidean = pd.DataFrame(haversine_distances_matrix)

# 엑셀 파일로 저장 (바탕 화면 경로)
df_euclidean.to_excel(desktop_path + 'haversine_euclidean_distances.xlsx', index=False)
