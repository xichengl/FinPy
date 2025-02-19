import numpy as np
import pandas as pd 
import re

class PanelResult:
    def __init__(self, result, reg_id='reg', digits=3, flag_star=True):
        self.result = result
        self.reg_id = reg_id
        self.digits = digits
        self.flag_star = flag_star
        self.output = self.parse_result()
    
    def format_values(self, series):
        return series.map(lambda s: f'{s:.{self.digits}f}')

    def create_significance_stars(self, pvalues):
        if self.flag_star:
            return np.where(pvalues <= 0.01, '^{***}', 
                            np.where(pvalues <= 0.05, '^{**}', 
                                    np.where(pvalues <= 0.1, '^{*}', '')))
        else:
            return np.where(pvalues <= 0.01, '***', 
                            np.where(pvalues <= 0.05, '**', 
                                    np.where(pvalues <= 0.1, '*', '')))
    
    def parse_result(self):
        beta = self.format_values(self.result.params.to_frame())
        t = self.format_values(self.result.tstats.to_frame())
        p = self.result.pvalues.to_frame()
        
        output = beta.join(t).join(p)
        
        output['sig'] = self.create_significance_stars(output['pvalue'])
        output['parameter'] = '$' + output['parameter'] + output['sig'] + '$'
        output['tstat'] = '($' + output['tstat'] + '$)'
        
        output = output.reset_index().rename(columns={'index': 'var'})
        _sort = output[['var']]
        _sort['order'] = range(len(_sort))
        
        output = output.melt(id_vars='var', value_vars=['parameter', 'tstat'],var_name='pt', value_name=self.reg_id).merge(_sort, on='var').sort_values(['order','pt']).drop(columns='order')
        
        add_info = pd.Series([
            f'{self.result.model.dependent.dataframe.columns[0]}', 
            f'{self.result.rsquared_within:.3f}', 
            f'{self.result.rsquared:.3f}', 
            f'{self.result.nobs:,.0f}'
        ])
        
        add_info = pd.DataFrame({
            'var': ['y', 'Within R2', 'R2', 'Observations'], 
            self.reg_id: add_info
        })
        
        output = pd.concat([output, add_info], axis=0).reset_index(drop=True)
        return output


def compare_panel_results(result_lst, order=[], label=[], digits=3, flag_star=True):
    ht_order = {}
    if len(order) > 0:
        ht_order = {j:i for i,j in enumerate(order)}
    else:
        order = PanelResult(result_lst[0]).output['var'].drop_duplicates().tolist()
        ht_order = {j:i for i,j in enumerate(order)}
    
    ht_order['y'] = -2
    ht_order['Intercept'] = -1
    ht_order['Within R2'] = 10000
    ht_order['R2'] = 10001
    ht_order['Observations'] = 10002
    
    output = None
    
    for i in range(len(result_lst)):
        res = PanelResult(result_lst[i], f'({i + 1})', digits=digits, flag_star=flag_star).output
        
        if output is None:
            output = res
        else:
            output = output.merge(res, on=['var', 'pt'], how='outer')
    
    
    output['order'] = output['var'].map(ht_order).fillna(9999)
    output = output.sort_values(['order', 'var', 'pt']).drop(columns=['pt', 'order']).reset_index(drop=True)

    # Mask certain variables
    output['var'] = output['var'].mask((output.index % 2 == 0) & ~(output['var'].isin(['y', 'Within R2', 'R2', 'Observations'])), '')

    # Apply custom labels if provided
    if len(order) > 0 and len(label) > 0:    
        label_dict = {i: j for i, j in zip(order, label)}
        output['var'] = output['var'].mask(output['var'].isin(label_dict.keys()), lambda s: s.map(label_dict))

    # Adjust formatting
    max_width = output['var'].str.len().max()
    output['var'] = output['var'].map(lambda x: x.ljust(max_width + 2))
    output.set_index('var', inplace=True)
    output.index.name = None
    output = output.fillna('')

    # Format the output for display
    for s in output.columns:
        output[s] = output[s].map(lambda x: x.ljust(20))

    return output


def output_reg(_resdata):
    width = _resdata.shape[1]
    soup = _resdata.to_latex()
    
    pattern_1 = r'\\begin\{tabular\}\{l+.*\}\n\\toprule'
    replacement_1 = r'\\begin{tabularx}{\\textwidth}{@{}  l @{\\extracolsep{\\fill}} ' + 'c'*width + r' @{}}\n\\midrule\\midrule'
    soup = re.sub(pattern_1, replacement_1, soup)
    
    pattern_2 = r'\\bottomrule\n\\end{tabular}'
    replacement_2 = r'\\midrule\n\\end{tabularx}'
    soup = re.sub(pattern_2, replacement_2, soup)
    return soup