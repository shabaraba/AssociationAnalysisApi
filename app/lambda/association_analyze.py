import pandas as pd
import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def handler(event, context):
    """
        request: json such as bellow:
            {
                data: [
                    ['aaa', 'bbb', 'ccc'],
                    ['aaa', 'ccc'],
                    ['bbb'],
                ],
                min_support: 0.01,
                rule: {
                    metric: "confidence",
                    min_threshold: "0.1"
                },
                condition: {
                    confidence: 0.2,
                    lift: 1.0
                }
            }
    """
    print(event)
    request = json.loads(event.get("body"))
    print(request)

    transactions = request.get('data')
    if not transactions:
        return { 'statusCode': 200, 'body': "[]" }

    # データをテーブル形式に加工
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # display(df)

    # itemsets・support算出
    freq_items = apriori(
        df,                                                # データフレーム
        min_support=float(request.get("min_support", 0.01)), # 支持度(support)の最小値
        use_colnames=True,                                 # 出力値のカラムに購入商品名を表示
        max_len=None,                                      # 生成されるitemsetsの個数
        verbose=1,                                         # low_memory=Trueの場合のイテレーション数
        low_memory=True,                                   # メモリ制限あり＆大規模なデータセット利用時に有効
    )
    sorted_freq_items = freq_items.sort_values(
        "support", ascending=False).reset_index(drop=True)
    print(sorted_freq_items)

    # アソシエーション・ルール抽出
    rule = request.get("rule")
    df_rules = association_rules(
        sorted_freq_items,                            # supportとitemsetsを持つデータフレーム
        metric=rule.get("metric", "confidence"),      # アソシエーション・ルールの評価指標
        min_threshold=float(rule.get("min_threshold", 0.1)), # metricsの閾値
    )
    print (df_rules)

    condition = request.get("condition")
    results = df_rules[
        (df_rules['confidence'] > float(condition.get("confidence", 0.2))) &  # 信頼度
        (df_rules['lift'] > float(condition.get("lift", 1.0)))  # リフト値
        & (df_rules['antecedents'].len == 1)
        & (df_rules['consequents'].len == 1)
    ]
    print(results.loc[:,["antecedents","consequents","confidence","lift"]])
    response = json.loads(results.to_json(orient="table")).get("data")
    print(json.dumps(response))

    return {
        'statusCode': 200,
        'body': json.dumps(response),
    }


if __name__ == '__main__':
    handler(event={
        "body": json.dumps({
            "data": [["nginx","powercmsx"],["Python","game","3DCG","animation","Blender"],["AWS","aws-cli"],["SageMaker","presigned-url","SageMakerStudio"],["Python","Julia","Filter","DataFrame"],["PHP","PHP7"],["React","State","Recoil"],["Python","watchdog","例外処理","フォルダ監視"],["Python","DeepLearning","Python3","RNN","PyTorch"],["BizRobo!"],["プロジェクトマネジメント"],["BizRobo!"],["データベース","BizRobo!"],["エラーハンドリング","BizRobo!"],["API","kintone","BizRobo!"],["初心者","iRIC","格子生成アルゴリズム"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["C#","Delphi","アルゴリズム","行列計算"],["neovim","NixOS","nix"],["Mac","VPN"],["Python","OpenCV","画像認識","プログラミング初心者","駆け出しエンジニア"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["nginx","websocket","Laravel","broadcast","RockyLinux"],["Blender","OpenFOAM","paraview"],["JavaScript","React"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["Python","ビットコイン","仮想通貨"],["アプリ開発","ノーコード","AppSheet"],["HTML","JavaScript","disabled"],["Webhook","Hexabase"],["C#","datetime","時刻計算","月末日","ADDMONTHS"],["R"],["JavaScript","TypeScript","crypto","PKCE"],["Git","SourceTree"],["test","テスト","testing","テスト自動化","ChatGPT"],["Go"],["Logix","NeosVR","fm変調"],["Node.js","サーバーレス","Hexabase","blastengine"],["am変調","Logix","NeosVR"],["アーキテクチャ","OREILLY","データ管理","ScaledArchitecture"],["Ruby","Rails","RSpec"],["Laravel","LaravelExcel"],["UiPath","UiPathStudio","UiPathFriends"],["SSH","SourceTree","MacbookPro"],["Hyper-V","コンテナ","WSL2"],["Dart","SDK","全文検索","Flutter","Hexabase"],["SystemVerilog","RISC-V"],["AWS","DMS"],["Windows","Git","GitHub","GithubDesktop"],["Node.js","サーバーレス","Hexabase","blastengine"],["AWS","Cloudtrail","GuardDuty","Macie","reinvent2022"],["PHP","Laravel"],["C#","VisualStudio","Windows10"],["C#","VisualStudio","Windows10"],["Docker"],["JavaScript","es6","資格","フロントエンド","HTML5プロフェッショナル認定試験対策"],["Python","VSCode","Pipenv"],["資格","TOGAF","TheOpenGroup","EnterpriseArchitecture"],["Bizrobo","BizRobo!"],["Bash","ポエム"],["Go","Line","GoogleCloudPlatform","gcp"],["Go","Line","GoogleCloudPlatform","gcp"],["Go","Line","GoogleCloudPlatform","gcp"],["WordPress","MySQL","nginx","Docker"],["HTTP","Web","authentication"],["Python"],["ポエム","転職"],["Python","Flask-WTF"],["Salesforce","GitLab","DevHub","SFDX","SalesforceCLI"],["oracle","Weblogic","OracleBI"],["gpt-3"],["Python","log","TOML","logging","Poetry"],["Azure","AzureCLI","AzureDevOps","AzureRepos"],["ClipStudioPaint"],["PowerApps","PowerPlatform"],["Android","Mac","Git","VSCode","M2"],["Google","プロジェクト管理","プロジェクトマネジメント"],["Python","install","pip","TensorFlow"],["Python","Django","gunicorn","ロリポップ"],["Network","ipsec","nat","l2tpv3","ヤマハルーター"],["JavaScript","clone","lodash","Deepcopy","structuredClone"],["Alpine.js"],["Ubuntu","Azure","オンプレミス","WSL"],["React"],["プログラミング","エンジニア","未経験エンジニア","プログラミング初心者","駆け出しエンジニア"],["nginx","jsonserver","Let’sEncrypt"],["Security","ニュース"],["C#","Xaml","enum","Generics","MAUI"],["Windows","PowerShell"],["Ruby","Rails","kaminari"],["アウトプット"],["PHP","for文","構文"],["PHP","php.ini","同時接続"],["初心者","入門"],["AWS","Unity","初心者","AR","ゲーム開発"],["Arduino","NewRelic","M5stack"],["Ubuntu","VirtualBox"],["PowerAutomate"],
                    ["Ubuntu","VirtualBox"],["PowerAutomate"],["C","Linux","Ubuntu","mind"],["Python","Windows","自動化","Slack"],["AI","画像生成","StableDiffusion"],["TypeScript","Generics"],["Git","GitHub"],["Qiita","初心者","記事作成"],["Ruby","Rails","VSCode"],["AWS"],["Node.js","Timestamp","ASN.1","jsrsasign"],["server","Ubuntu","minecraft","Docker","docker-compose"],["Android","XML","tips","AndroidStudio","BuildVariant"],["WordPress","PHP8"],["Ruby","Rails","VSCode"],["Git"],["Motionbuilder","mocopi"],["JavaScript"],["SQL","PostgreSQL"],["VBScript","WinActor"],["Zsh","初学者"],["Python","DeepLearning","Python3","DNN"],["画像生成","StableDiffusion","dreambooth"],["Swift"],["AWS","IBM","ibmcloud"],["Python","SQL","PostgreSQL"],["ShellScript","Bash","国際化","Gettext","POSIX"],["ShinobiLayer","ibmcloud"],["Arrow","ApacheArrow"],["Ruby","初心者","初心者向け"],["React","学習記録","Next.js"],["React"],["WordPress","mysql5.7","MariaDB10.3","PHP7.4"],["Python","pandas","query関数"],["YARN"],["AI","ChatGPT"],["Laravel","SPA"],["ffmpeg","RaspberryPi","カメラ","HLS"],["Terminal","Python3","ペイント"],["Ubuntu"],["SwiftUI","iOS16"],["AWS"],["英語","参考文献","AUTOSAR","小川メソッド","CoutdownCalendar2022"],["redhat","rpm"],["Python","brew","venv","M1","StableDiffusion"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["HTML","CSS","JavaScript"],["Python","機械学習","分類分析"],["API","Vue.js","axios"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["AWS"],["テスト","インターン","大学生","交流会","テストエンジニア"],["React","学習記録"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["Salesforce","MarketingCloud","ローコード","JourneyBuilder","AutomationStudio"],["SSH","Docker","サーバー","備忘録","Xサーバー"],["AWS","S3","lambda","EventBridge"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["AWS","S3","aws-cli","CloudShell"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["ゲーム開発","CocosCreator","GoogleAdsense","WebGPU","プラットホーム"],["Julia","Query","tidyverse","Polars"],["Python","XML","JSON","Vue.js","デジタル庁"],["TypeScript","React","Storage","Next.js","Supabase"],["Python","AWS","pip","cfn-lint"],["HTML"],["HTML"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["Erlang","Elixir","AdventCalendar2023","闘魂","アドハラ"],["PHP","初心者"],["初心者","CAPTCHA"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2022"],["Go"],["Ruby","初学者向け","初学者"],["C#","Xaml","WinUI3"],["CI","GitHubActions"],["Security","初心者","サイバー攻撃","ブルートフォース","リバースブルートフォース"],["英語","参考文献","AUTOSAR","小川メソッド","CountdownCalendar2023"],["Python","初心者","DeepLearning"],["ShellScript","Haskell","Rust","チューリング完全","BitwiseCyclicTag"],["MySQLWorkbench"],["Python","AtCoder","競技プログラミング","競プロ"],["ServiceNow"],["RAM","GoogleColaboratory"],["Windows","コマンド","バッチ処理"],["MongoDB","Wekan","docekr"],["AWS","Amazon","CloudWatch","ECS","AWSControlTower"],["AWS","SSM"],["Java"],["Rails","MySQL","全文検索"],["Java"],["JavaScript","Atlassian","React"],["Ansible"],["Java"],["Google","プロジェクト管理","プロジェクトマネジメント"],["Java"],["table","配列","PHP7","empty","array_filter"],["Ruby","JavaScript","Rails","React"],
                    ["Java"],["C++","アルゴリズム","AtCoder","初心者","競技プログラミング"],["Android","iOS","Flutter","SingleChildScrollView","showDialog"],["Vue.js","Nuxt3"],["Google","プロジェクト管理","プロジェクトマネジメント"],["Android","Kotlin","JetpackCompose"],["Windows","Salesforce","Apex"],["AWS"],["sandbox","akamai","Edgeworkers","akamai-cli","edgeworkers-cli"],["Python","Tkinter","pyserial"],["Google","プロジェクト管理","プロジェクトマネジメント"],["Excel","SharePoint","SharePointOnline","PowerAutomate"],["create","Laravel","update","delete","クエリビルダ"],["iOS","初心者","UIKit","Swift","大学生"],["WebAPI","JSON","Laravel"],["roomba","LQR"],["Java","Android","AndroidStudio","android開発","AndroidTV"],["Google","プロジェクト管理","プロジェクトマネジメント"],["数理最適化"],["C#"],["Node.js","TypeScript","GraphQL","n+1問題"],["connected-home-ip"],["AWS","aws-cli","CloudShell"],["RaspberryPi","ラズパイ"],["TypeScript","React"],["Python","OBS"],["Google","プロジェクト管理","プロジェクトマネジメント"],["Ubuntu","Python3","VisualStudioCode"],["Java","Android","Kotlin","AndroidStudio","toolbar"],["Vim","Mac","Eclipse"],["ansible-playbook"],["Node.js","AWS","lambda"],["Bash","Linux","AWS","EC2"],["Python","brew","edit"],["プロジェクト管理","エンジニアリング"],["学生","大学生","就活","専門学校"],["PHP","Laravel"],["JavaScript","Rails","Rails5"],["SQL"],["WPF","dictionary","Xaml","Generics","MAUI"],["Google","プロジェクト管理","プロジェクトマネジメント"],["React"],["gcp","Firebase","Next.js","GitHubActions"],["oracle","analytics","ビジュアライゼーション","データセット"],["GitHub","pullrequest","GitHubActions"],["Android","Kotlin","JetpackCompose"],["JavaScript","TypeScript","React"],["IBM","cp4ba","FileNet","ViewONE"],["Apache","TypeScript","Fresh","deno"],["Auzre","SynapseAnalytics"],["議事録","会議","ミーティング"],["SQLite","serverless","CloudflareWorkers","ChatGPT","CloudflareD1"],["Python","Flask","SQLite"],["Dataverse"],["Docker","ECS","CodePipeline","ECR","CICD"],["Databricks","UnityCatalog"],["Django","form","フォーム","取得"],["Ruby","初学者向け","初学者"],["Flask"],["C#"],["PHP","MySQL","Bootstrap","Laravel","Docker"],["Adobe","AdobeExperienceCloud","AdobeCampaign"],["ModSecurity","kubernetes","CRS","ingress-nginx"],["Go"],["SpringBoot","VSCode","環境構築手順"],["AWS","EventBridge"],["Logix","NeosVR"],["REST-API","Insomnia"],["Chrome"],["HTTP","Web","Let’sEncrypt","Caddy"],["Azure","SynapseAnalytics"],["PHP","API","Line","linebot","LINEmessagingAPI"],["Excel","日付","PowerBI","PowerQuery"],["Git","AWS","CodeCommit"],["JavaScript","レベルアップ問題集","paizaラーニング","二分探索メニュー応用編"],["Python","Ubuntu","#chromium","#Webスクレイピング","#GoogleColab"],["IDE","VSCode","Databricks"],["Go"],["Microsoft","Azure","monitoring","負荷試験","監視"],["SAML","kintone","GoogleWorkspace"],["AWS","SSM"],["UnrealEngine","VRM","ue5","#VRM4U"],["Java","Windows"],["Java","Tomcat","SpringBoot","VSCode"],["oracle","analytics","ビジュアライゼーション","データセット"],["PowerBI","PowerPlatform","Microsoft365"],["PowerBI","PowerPlatform","Microsoft365"],["PowerBI","PowerPlatform","Microsoft365"],["PowerBI","PowerPlatform","Microsoft365"],["GitHub","shell","GH"],["Android","Kotlin"],["機械学習","MachineLearning","AI","人工知能","データサイエンス"],["RaspberryPi"],["Network","インフラ","ルーター"],["Laravel","REPL","Tinker","namespace"],["API","GAS","porters"],["Elixir","データ分析","Explorer","統計学","nx"],["Git"],["PHP","HTML5"],["Python","競プロ"]
            ],
            "min_support": 0.04,
            "rule": {
                "metric": "confidence",
                "min_threshold": 0.5
            },
            "condition": {
                "confidence": 0.5,
                "lift": 5
            }
        })
    }, context='')
