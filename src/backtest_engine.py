import numpy as np
import pandas as pd

class IronCondorBacktester:
    def __init__(self, strike_interval=50):
        self.strike_interval = strike_interval

    def construct_iron_condor(self, S_star, current_chain):
        """
        Determines the 4 legs of the Iron Condor based on the attractor S*
        and the market-implied escape probability (Straddle Premium).
        """
        # 1. Find the Center Strike (Nearest to S*)
        center_strike = round(S_star / self.strike_interval) * self.strike_interval
        
        # 2. Get the prices of the Call and Put at the Center Strike
        # We use a try-except block just in case of missing data in the CSV
        try:
            short_call = current_chain[(current_chain['Strike'] == center_strike) & 
                                       (current_chain['Option_Type'] == 'CE')].iloc[0]
            
            short_put = current_chain[(current_chain['Strike'] == center_strike) & 
                                      (current_chain['Option_Type'] == 'PE')].iloc[0]
        except IndexError:
            return None # Skip this timestamp if data is missing
            
        # 3. Calculate 1-Sigma Escape Bound (Straddle Premium)
        straddle_premium = short_call['Close'] + short_put['Close']
        
        # 4. Determine Wing Strikes (Center +/- Implied Move)
        upper_wing_raw = center_strike + straddle_premium
        lower_wing_raw = center_strike - straddle_premium
        
        long_call_strike = round(upper_wing_raw / self.strike_interval) * self.strike_interval
        long_put_strike = round(lower_wing_raw / self.strike_interval) * self.strike_interval
        
        # 5. Get the prices of the Wings
        try:
            long_call = current_chain[(current_chain['Strike'] == long_call_strike) & 
                                      (current_chain['Option_Type'] == 'CE')].iloc[0]
            
            long_put = current_chain[(current_chain['Strike'] == long_put_strike) & 
                                     (current_chain['Option_Type'] == 'PE')].iloc[0]
        except IndexError:
            return None

        # 6. Calculate Net Credit Collected
        net_credit = (short_call['Close'] + short_put['Close']) - (long_call['Close'] + long_put['Close'])
        
        # Return the structured trade
        trade_setup = {
            'Center_Strike': center_strike,
            'Short_Call_Price': short_call['Close'],
            'Short_Put_Price': short_put['Close'],
            'Long_Call_Strike': long_call_strike,
            'Long_Call_Price': long_call['Close'],
            'Long_Put_Strike': long_put_strike,
            'Long_Put_Price': long_put['Close'],
            'Net_Credit_Collected': net_credit,
            'Max_Risk': (long_call_strike - center_strike) - net_credit
        }
        
        return trade_setup