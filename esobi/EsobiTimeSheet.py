#!/usr/bin/env python
import pandas as pd


def execute():
    # 1. 取得time_sheet_path
    time_sheet_path = input('請輸入TimeSheet位置')

    # 2. 取得檔案內容
    time_sheet = pd.read_excel(time_sheet_path)
    export_sheet = pd.DataFrame()
    export_sheet['member'], export_sheet['time'] = time_sheet['登記人'], time_sheet['耗時']
    export_sheet = export_sheet.groupby(['member']).sum()

    # 3. 取得member_path
    member_path = input('請輸入人員位置')

    # 4. 篩選出今日上班人員
    members = pd.read_excel(member_path)
    members = members[members.check == 'Y'].drop('check', axis=1)

    # 5. 排除重複輸入的人員
    members = members.drop_duplicates(keep=False)

    # 6. join 人員清單與工時
    final_data = pd.merge(members, export_sheet, how='left', on='member')
    final_data.fillna(0, inplace=True)
    final_data = final_data[final_data.time < 8]
    print(final_data)


if __name__ == '__main__':
    execute()
