import streamlit as st
import requests
import json

# 设置页面标题
st.set_page_config(page_title="睡眠障碍预测系统", page_icon="🌙")

st.title("🌙 睡眠障碍智能诊断系统")
st.markdown("请输入您的身体指标，模型将预测潜在的睡眠问题。")

# --- 1. 左侧侧边栏：输入表单 ---
with st.sidebar:
    st.header("📋 患者信息录入")
    
    gender = st.selectbox("性别", ["Male", "Female"])
    age = st.slider("年龄", 10, 90, 32)
    occupation = st.selectbox("职业", [
        "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
        "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
        "Salesperson", "Manager"
    ])
    sleep_duration = st.slider("睡眠时长 (小时)", 4.0, 10.0, 7.0, 0.1)
    quality_of_sleep = st.slider("睡眠质量 (1-10)", 1, 10, 7)
    physical_activity = st.slider("体力活动水平 (分钟/天)", 0, 100, 40)
    stress_level = st.slider("压力等级 (1-10)", 1, 10, 5)
    bmi_category = st.selectbox("BMI 类别", ["Normal", "Overweight", "Obese"])
    blood_pressure = st.text_input("血压 (例如 120/80)", "120/80")
    heart_rate = st.number_input("心率 (bpm)", 60, 120, 70)
    daily_steps = st.number_input("每日步数", 0, 20000, 5000)

    # 构造发送给 API 的数据字典
    input_data = {
        "gender": gender,
        "age": age,
        "occupation": occupation,
        "sleep_duration": sleep_duration,
        "quality_of_sleep": quality_of_sleep,
        "physical_activity_level": physical_activity,
        "stress_level": stress_level,
        "bmi_category": bmi_category,
        "blood_pressure": blood_pressure,
        "heart_rate": heart_rate,
        "daily_steps": daily_steps
    }

# --- 2. 主页面：预测按钮与结果展示 ---
if st.button("🚀 开始预测", type="primary"):
    with st.spinner("模型正在分析数据..."):
        try:
            # 这里的 localhost 指的是容器内部，Streamlit 访问同容器内的 FastAPI
            # 注意：在生产环境中，这通常指向 API 的服务名，但在单容器里 localhost 是通的
            api_url = "http://127.0.0.1:8000/invocations" 
            
            response = requests.post(api_url, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "未知")
                
                st.success("✅ 预测完成！")
                
                # 美化结果展示
                st.subheader(f"诊断结果: {prediction}")
                
                if prediction == "None":
                    st.info("恭喜！未检测到明显的睡眠障碍风险。保持健康的生活习惯！Data from: Model")
                elif prediction == "Insomnia":
                    st.warning("⚠️ 警告：检测到失眠症 (Insomnia) 风险。建议咨询医生或改善作息。")
                elif prediction == "Sleep Apnea":
                    st.error("🚨 警告：检测到睡眠呼吸暂停 (Sleep Apnea) 风险。请尽快就医检查。")
            else:
                st.error(f"❌ 预测失败: {response.text}")
                
        except Exception as e:
            st.error(f"❌ 无法连接到后端服务: {e}")

# 页脚
st.markdown("---")
st.caption("Powered by MLOps Pipeline & Streamlit")