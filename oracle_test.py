import oracledb

# 1. 오라클 데이터베이스 접속 정보 설정
db_config = {
    "user": "YOUR_USERNAME",  # 사용자명 입력
    "password": "YOUR_PASSWORD",  # 비밀번호 입력
    "dsn": "localhost:1521/xe"  # 접속 주소 (호스트:포트/서비스이름)
}


def setup_table_if_not_exists(cursor):
    """테이블이 존재하는지 확인하고 없으면 생성합니다."""
    # USER_TABLES 뷰를 조회하여 EMP 테이블이 있는지 확인 (테이블명은 대문자로 조회)
    cursor.execute("SELECT count(*) FROM user_tables WHERE table_name = 'EMP'")
    (table_exists,) = cursor.fetchone()

    if table_exists == 0:
        print("💡 EMP 테이블이 존재하지 않습니다. 새로 생성합니다...")
        create_table_sql = """
        CREATE TABLE EMP (
            id VARCHAR2(50) PRIMARY KEY,
            name VARCHAR2(100) NOT NULL,
            address VARCHAR2(255),
            phone1 VARCHAR2(20),
            phone2 VARCHAR2(20),
            emp_info CLOB,
            reg_date DATE DEFAULT SYSDATE
        )
        """
        cursor.execute(create_table_sql)
        print("✅ EMP 테이블 생성이 완료되었습니다.")
    else:
        print("💡 EMP 테이블이 이미 존재합니다. 기존 테이블을 사용합니다.")


def insert_emp_data(cursor, connection, emp_data):
    """EMP 테이블에 사원 정보를 등록합니다."""
    # 바인드 변수(:1, :2 등)를 사용하여 SQL Injection 방지 및 CLOB 데이터 안전 처리
    insert_sql = """
    INSERT INTO EMP (id, name, address, phone1, phone2, emp_info, reg_date)
    VALUES (:1, :2, :3, :4, :5, :6, SYSDATE)
    """
    try:
        cursor.execute(insert_sql, emp_data)
        connection.commit()  # 변경사항 저장
        print(f"✅ 사원 '{emp_data[1]}'님의 정보가 성공적으로 등록되었습니다.")
    except oracledb.IntegrityError as e:
        print(f"❌ 데이터 등록 실패: 기본키(ID) 중복 등의 무결성 에러 발생 - {e}")
    except Exception as e:
        print(f"❌ 데이터 등록 중 에러 발생: {e}")


# 메인 실행 로직
def main():
    connection = None
    try:
        # DB 접속
        connection = oracledb.connect(**db_config)
        cursor = connection.cursor()

        # 1. 테이블 생성 또는 확인
        setup_table_if_not_exists(cursor)

        # 2. 등록할 샘플 데이터 준비 (emp_info에 들어갈 아주 긴 CLOB용 텍스트)
        long_clob_text = "이 사원은 파이썬 oracledb 라이브러리를 통해 등록되었습니다. \n" + \
                         ("CLOB 컬럼 테스트를 위한 아주 긴 텍스트 데이터입니다. " * 50)

        # id, name, address, phone1, phone2, emp_info 순서
        new_employee = (
            'EMP20231001',
            '홍길동',
            '서울특별시 강남구 테헤란로',
            '010-1234-5678',
            '02-987-6543',
            long_clob_text
        )

        # 3. 데이터 등록 실행
        insert_emp_data(cursor, connection, new_employee)

        # 4. (선택) 데이터가 잘 들어갔는지 CLOB 포함하여 확인 (조회 시 CLOB을 문자열로 자동 변환)
        oracledb.defaults.fetch_lobs = False  # LOB 객체 대신 일반 Python 문자열로 가져오기 위한 설정

        cursor.execute("SELECT id, name, emp_info FROM EMP WHERE id = :1", ['EMP20231001'])
        row = cursor.fetchone()
        if row:
            print("\n🔍 [조회 테스트]")
            print(f"ID: {row[0]}, Name: {row[1]}")
            print(f"CLOB Data (일부): {row[2][:100]}...")  # 긴 데이터이므로 100자만 출력

    except oracledb.DatabaseError as e:
        print(f"데이터베이스 연결/실행 에러: {e}")

    finally:
        # 자원 해제
        if 'cursor' in locals() and cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    main()