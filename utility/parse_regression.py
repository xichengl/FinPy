import numpy as np
import pandas as pd 
import re

class FeolsResult:
    def __init__(self, result, reg_id = 'reg', digits = 3):
        self.result = result
        self.reg_id = reg_id
        self.digits = digits
        self.output = self.parse_result()

    def format_values(self, series):
        return series.map(lambda s: f'{s:.{self.digits}f}')
    
    def starwork(self, pvalues):
        return np.where(pvalues <= 0.01, '^{***}', 
                        np.where(pvalues <= 0.05, '^{**}', 
                                np.where(pvalues <= 0.1, '^{*}', '')))
    
    def parse_result(self):
        coef = self.format_values(self.result.coef().to_frame().rename(columns = {'Estimate': 'coef'}))
        tstat = self.format_values(self.result.tstat().to_frame().rename(columns = {'t value': 'tstat'}))
        pvalue = self.result.pvalue().to_frame().rename(columns = {'Pr(>|t|)': 'pvalue'})
        
        output = coef.join(tstat).join(pvalue)
        output['sig'] = self.starwork(output['pvalue'])
        output['coef'] = '$' + output['coef'] + output['sig'] + '$'
        output['tstat'] = '($' + output['tstat'] + '$)'
        
        output = output.reset_index().rename(columns = {'Coefficient': 'exog'})
        _order = output[['exog']]
        _order['order'] = range(len(_order))
        
        output = output.melt(id_vars = 'exog', value_vars = ['coef', 'tstat'], var_name = 'ct', value_name = self.reg_id).merge(_order, on = 'exog').sort_values(['order', 'ct']).drop(columns = 'order')
        
        if self.result._has_fixef:
            add_info = pd.DataFrame({
                'exog': ['endog', 'Within R2', 'R2', 'Observations'],
                self.reg_id: pd.Series([
                    f'{self.result._model_name.split('~')[0]}',
                    f'{self.result._r2_adj_within:.3f}',
                    f'{self.result._r2:.3f}',
                    f'{self.result._N_rows:,.0f}'
                ])            
            })

        else:
            add_info = pd.DataFrame({
                'exog': ['endog', 'R2', 'Observations'],
                self.reg_id: pd.Series([
                    f'{self.result._model_name.split('~')[0]}',
                    f'{self.result._r2:.3f}',
                    f'{self.result._N_rows:,.0f}'
                ])            
            })

        output = pd.concat([output, add_info], axis = 0).reset_index(drop = True)
        return output

def compare_feols_results(result_lst, order = [], label = [], digits = 3):
    ht_order = {}
    if len(order) > 0:
        ht_order = {j: i for i, j in enumerate(order)}
    else:
        order = []
        for i in result_lst:
            exog = FeolsResult(i).output['exog'].drop_duplicates().tolist()
            for var in exog:
                if var not in order:
                    order.append(var)        
        ht_order = {j: i for i, j in enumerate(order)}
    
    ht_order['endog'] = -2
    ht_order['Intercept'] = -1
    ht_order['Within R2'] = len(order) + 1
    ht_order['R2'] = len(order) + 2
    ht_order['Observations'] = len(order) + 3
    
    output = None
    
    for i in range(len(result_lst)):
        res = FeolsResult(result_lst[i], f'({i + 1})', digits = digits).output
        
        if output is None:
            output = res
        else:
            output = output.merge(res, on = ['exog', 'ct'], how='outer')
    
    
    output['order'] = output['exog'].map(ht_order).fillna(9999)
    output = output.sort_values(['order','exog','ct']).drop(columns = ['ct','order']).reset_index(drop = True)

    # Mask certain variables
    output['exog'] = output['exog'].mask((output.index % 2 == 0) & ~(output['exog'].isin(['endog','Within R2','R2','Observations'])), '')

    # Apply custom labels if provided
    if len(order) > 0 and len(label) > 0:    
        label_dict = {i: j for i, j in zip(order, label)}
        output['exog'] = output['exog'].mask(output['exog'].isin(label_dict.keys()), lambda s: s.map(label_dict))

    # Adjust formatting
    max_width = output['exog'].str.len().max()
    output['exog'] = output['exog'].map(lambda x: x.ljust(max_width + 2))
    output.set_index('exog', inplace = True)
    output.index.name = None
    output = output.fillna('')

    # Format the output for display
    for s in output.columns:
        output[s] = output[s].map(lambda x: x.ljust(20))

    return output


def texize(_res):
    width = _res.shape[1]
    soup = _res.to_latex()
    
    pattern_1 = r'\\begin\{tabular\}\{l+.*\}\n\\toprule'
    replacement_1 = r'\\begin{tabularx}{\\textwidth}{@{}  l @{\\extracolsep{\\fill}} ' + 'c'*width + r' @{}}\n\\midrule\\midrule'
    soup = re.sub(pattern_1, replacement_1, soup)
    
    pattern_2 = r'\\bottomrule\n\\end{tabular}'
    replacement_2 = r'\\midrule\n\\end{tabularx}'
    soup = re.sub(pattern_2, replacement_2, soup)
    return soup

