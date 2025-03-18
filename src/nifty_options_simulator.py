import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import date
from dataclasses import dataclass
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
os.environ["QT_QPA_PLATFORM"] = "windows:fonts"
import seaborn as sns
import json
import time  # Add time module for execution timing

@dataclass
class Trade:
    entry_time: datetime
    option_type: str  # 'CALL' or 'PUT'
    strike_price: float
    entry_price: float
    stop_loss: float
    target: float
    status: str = 'ACTIVE'  # 'ACTIVE', 'SL_HIT', 'TARGET_HIT'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    volume_5min: Optional[float] = None
    sentiment: Optional[float] = None
    entry_spot: Optional[float] = None
    exit_spot: Optional[float] = None
    price_change: Optional[float] = None
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None

class NiftyOptionsSimulator:
    def __init__(self, spot_file: str, futures_file: str, sentiment_file: str, oi_file: str):
        # Load and preprocess data
        self.spot_data = pd.read_csv(spot_file)
        self.futures_data = pd.read_csv(futures_file)
        self.sentiment_data = pd.read_csv(sentiment_file)
        self.oi_data = pd.read_csv(oi_file)
        
        # Convert datetime columns
        self.spot_data['datetime'] = pd.to_datetime(self.spot_data['datetime'])
        self.futures_data['datetime'] = pd.to_datetime(self.futures_data['datetime'])
        self.sentiment_data['datetime'] = pd.to_datetime(self.sentiment_data['datetime'])
        self.oi_data['datetime'] = pd.to_datetime(self.oi_data['datetime'])
        
        # Preprocess 5-minute volume data
        self.futures_data['5min_group'] = self.futures_data['datetime'].dt.floor('5min')
        self.volume_5min = self.futures_data.groupby('5min_group')['volume'].sum().reset_index()
        self.volume_5min.columns = ['datetime', 'volume_5min']
        
        # Debug information
        print("\nInitial data summary:")
        print(f"Spot data rows: {len(self.spot_data)}")
        print(f"Futures data rows: {len(self.futures_data)}")
        print(f"Sentiment data rows: {len(self.sentiment_data)}")
        print(f"OI data rows: {len(self.oi_data)}")
        print("\n5-minute volume aggregation:")
        print(self.volume_5min.head())
        
        # Initialize trade tracking
        self.active_trades: List[Trade] = []
        self.completed_trades: List[Trade] = []
        self.target_level = 22400
        self.pending_trade = None  # Store trade signal for next minute entry
        
    def get_5min_volume(self, timestamp: datetime) -> float:
        """Get the 5-minute volume for the given timestamp"""
        five_min_ts = timestamp.floor('5min')
        volume = self.volume_5min[self.volume_5min['datetime'] == five_min_ts]['volume_5min']
        return volume.iloc[0] if not volume.empty else 0
    
    def get_sentiment(self, timestamp: datetime) -> Optional[float]:
        """Get the sentiment value for the given timestamp"""
        sentiment = self.sentiment_data[self.sentiment_data['datetime'] == timestamp]['high_sent']
        return sentiment.iloc[0] if not sentiment.empty and not pd.isna(sentiment.iloc[0]) else None
        
    def get_atm_strike(self, spot_price: float, option_type: str) -> float:
        """Calculate ATM strike price based on spot price and option type"""
        strike_interval = 50
        if option_type == 'CALL':
            return strike_interval * (spot_price // strike_interval)
        else:  # PUT
            return strike_interval * ((spot_price + strike_interval - 1) // strike_interval)
    
    def check_entry_condition(self, current_row: pd.Series, prev_row: pd.Series, current_time: datetime) -> tuple[Optional[str], Optional[str]]:
        """Check if entry conditions are met"""
        # Check price change condition
        price_change = current_row['Close'] - prev_row['Close']
        threshold = 50 * 0.55  # 55% of index strike price interval
        
        # Debug information
        print(f"\nChecking conditions for {current_time}:")
        print(f"Price change: {price_change:.2f}, Threshold: {threshold:.2f}")
        
        if abs(price_change) <= threshold:
            print("Price change condition not met")
            return None, None
            
        # Get additional data
        volume_5min = self.get_5min_volume(current_time)
        sentiment = self.get_sentiment(current_time)
        
        print(f"5-min Volume: {volume_5min}")
        print(f"Sentiment: {sentiment}")
        
        entry_reasons = []
        
        # Check conditions and build entry reason
        if abs(price_change) > threshold:
            entry_reasons.append(f"Price change: {price_change:.2f}")
        
        if volume_5min > 125000:
            entry_reasons.append(f"High volume: {volume_5min}")
            
        if sentiment is not None and abs(sentiment) >= 4:
            entry_reasons.append(f"Strong sentiment: {sentiment}")
        
        # All conditions met or data not available
        if entry_reasons:
            signal = 'CALL' if price_change > 0 else 'PUT'
            reason = " | ".join(entry_reasons)
            print(f"Signal generated: {signal}")
            return signal, reason
        return None, None
    
    def get_option_price(self, timestamp: datetime, strike: float, option_type: str) -> Optional[float]:
        """Get option price from OI data"""
        mask = (self.oi_data['datetime'] == timestamp) & \
               (self.oi_data['strike_price'] == strike) & \
               (self.oi_data['right'] == ('Call' if option_type == 'CALL' else 'Put'))
        if mask.any():
            return self.oi_data[mask]['Close'].iloc[0]
        return None
    
    def process_minute(self, current_time: datetime, current_row: pd.Series, prev_row: pd.Series):
        """Process each minute's data and manage trades"""
        # Execute pending trade if exists
        if self.pending_trade is not None:
            option_price = self.get_option_price(current_time, self.pending_trade['strike'], self.pending_trade['option_type'])
            if option_price is not None:
                new_trade = Trade(
                    entry_time=current_time,
                    option_type=self.pending_trade['option_type'],
                    strike_price=self.pending_trade['strike'],
                    entry_price=option_price,
                    stop_loss=self.pending_trade['stop_loss'],
                    target=self.target_level,
                    volume_5min=self.pending_trade['volume_5min'],
                    sentiment=self.pending_trade['sentiment'],
                    entry_spot=current_row['Close'],
                    price_change=self.pending_trade['price_change'],
                    entry_reason=self.pending_trade['entry_reason']
                )
                self.active_trades.append(new_trade)
            self.pending_trade = None
        
        # Check and update existing trades
        for trade in self.active_trades[:]:
            current_price = self.get_option_price(current_time, trade.strike_price, trade.option_type)
            if current_price is None:
                continue
                
            # Check target
            if current_row['Close'] <= self.target_level:
                trade.status = 'TARGET_HIT'
                trade.exit_time = current_time
                trade.exit_price = current_price
                trade.exit_spot = current_row['Close']
                trade.pnl = current_price - trade.entry_price
                trade.exit_reason = f"Target hit at {self.target_level}"
                self.active_trades.remove(trade)
                self.completed_trades.append(trade)
                continue
                
            # Check stop loss
            if (trade.option_type == 'CALL' and current_row['Low'] <= trade.stop_loss) or \
               (trade.option_type == 'PUT' and current_row['High'] >= trade.stop_loss):
                trade.status = 'SL_HIT'
                trade.exit_time = current_time
                trade.exit_price = current_price
                trade.exit_spot = current_row['Close']
                trade.pnl = current_price - trade.entry_price
                trade.exit_reason = f"Stop loss hit at {trade.stop_loss}"
                self.active_trades.remove(trade)
                self.completed_trades.append(trade)
        
        # Check for new trade entry signal
        entry_signal, entry_reason = self.check_entry_condition(current_row, prev_row, current_time)
        if entry_signal:
            atm_strike = self.get_atm_strike(current_row['Close'], entry_signal)
            stop_loss = current_row['High'] if entry_signal == 'PUT' else current_row['Low']
            volume_5min = self.get_5min_volume(current_time)
            sentiment = self.get_sentiment(current_time)
            price_change = current_row['Close'] - prev_row['Close']
            
            # Store trade details for execution in next minute
            self.pending_trade = {
                'option_type': entry_signal,
                'strike': atm_strike,
                'stop_loss': stop_loss,
                'volume_5min': volume_5min,
                'sentiment': sentiment,
                'price_change': price_change,
                'entry_reason': entry_reason
            }
    
    def plot_trades(self):
        """Plot trade entry/exit points and spot price movement with interactive features"""
        if not self.completed_trades:
            print("No trades to plot")
            return
            
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(rows=2, cols=1, 
                               vertical_spacing=0.15,
                               subplot_titles=('Spot Price and Trade Points', 'Trade PnL'))
    
            # Plot spot price
            fig.add_trace(
                go.Scatter(
                    x=self.spot_data['datetime'],
                    y=self.spot_data['Close'],
                    name='Spot Price',
                    line=dict(color='blue', width=2),
                    hovertemplate='Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
    
            # Plot trade entry and exit points
            for trade in self.completed_trades:
                color = 'green' if trade.option_type == 'CALL' else 'red'
                
                # Entry point
                entry_hover_text = "<br>".join([
                    f"Entry Time: {trade.entry_time}",
                    f"Type: {trade.option_type}",
                    f"Strike: {trade.strike_price}",
                    f"Entry Price: {trade.entry_price:.2f}",
                    f"Entry Spot: {trade.entry_spot:.2f}",
                    f"Volume (5min): {format(int(trade.volume_5min), ',') if trade.volume_5min is not None else 'N/A'}",
                    f"Sentiment: {f'{trade.sentiment:.2f}' if trade.sentiment is not None else 'N/A'}",
                    f"Price Change: {f'{trade.price_change:.2f}' if trade.price_change is not None else 'N/A'}",
                    f"Entry Reason: {trade.entry_reason if trade.entry_reason is not None else 'N/A'}"
                ])
                
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[trade.entry_spot],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if trade.option_type == 'CALL' else 'triangle-down',
                            size=15,
                            color=color
                        ),
                        name=f"{trade.option_type} Entry",
                        hovertemplate=entry_hover_text + '<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Exit point
                exit_hover_text = "<br>".join([
                    f"Exit Time: {trade.exit_time if trade.exit_time is not None else 'N/A'}",
                    f"Type: {trade.option_type}",
                    f"Strike: {trade.strike_price}",
                    f"Exit Price: {f'{trade.exit_price:.2f}' if trade.exit_price is not None else 'N/A'}",
                    f"Exit Spot: {f'{trade.exit_spot:.2f}' if trade.exit_spot is not None else 'N/A'}",
                    f"PnL: {f'{trade.pnl:.2f}' if trade.pnl is not None else 'N/A'}",
                    f"Status: {trade.status}",
                    f"Exit Reason: {trade.exit_reason if trade.exit_reason is not None else 'N/A'}"
                ])
                
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_spot],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=15,
                            color=color
                        ),
                        name=f"{trade.option_type} Exit",
                        hovertemplate=exit_hover_text + '<extra></extra>'
                    ),
                    row=1, col=1
                )
    
            # Plot PnL
            trade_times = [t.exit_time for t in self.completed_trades]
            pnls = [t.pnl for t in self.completed_trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
            
            hover_text = [
                "<br>".join([
                    f"Exit Time: {t.exit_time if t.exit_time is not None else 'N/A'}",
                    f"Type: {t.option_type}",
                    f"PnL: {f'{t.pnl:.2f}' if t.pnl is not None else 'N/A'}",
                    f"Status: {t.status}"
                ])
                for t in self.completed_trades
            ]
            
            fig.add_trace(
                go.Bar(
                    x=trade_times,
                    y=pnls,
                    marker_color=colors,
                    name='PnL',
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_text
                ),
                row=2, col=1
            )
    
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Trading Activity and Performance",
                title_x=0.5,
                title_font=dict(size=20),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                hovermode='x unified'
            )
    
            # Update axes
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="PnL", row=2, col=1)
    
            # Try multiple display options
            try:
                # First try regular show
                fig.show()
            except Exception as e1:
                print(f"Standard display failed, trying alternative methods... Error: {e1}")
                try:
                    # Try with renderer
                    fig.show(renderer="browser")
                except Exception as e2:
                    print(f"Browser display failed... Error: {e2}")
                    try:
                        # Save as HTML file
                        fig.write_html("trading_visualization.html")
                        print("Plot saved as 'trading_visualization.html'. Please open it in your web browser.")
                    except Exception as e3:
                        print(f"Failed to save HTML file... Error: {e3}")
                        
        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Falling back to basic statistics display...")
            # Display basic trade information
            for i, trade in enumerate(self.completed_trades, 1):
                print(f"\nTrade {i}:")
                print(f"Entry Time: {trade.entry_time}")
                print(f"Exit Time: {trade.exit_time}")
                print(f"Type: {trade.option_type}")
                print(f"PnL: {trade.pnl:.2f}")
                print(f"Status: {trade.status}")
    
    def run_simulation(self):
        """Run the complete simulation"""
        for i in range(1, len(self.spot_data)):
            current_row = self.spot_data.iloc[i]
            prev_row = self.spot_data.iloc[i-1]
            current_time = current_row['datetime']
            
            self.process_minute(current_time, current_row, prev_row)
    
    def get_simulation_results(self) -> pd.DataFrame:
        """Get simulation results as a DataFrame"""
        results = []
        for trade in self.completed_trades:
            results.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'option_type': trade.option_type,
                'strike_price': trade.strike_price,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'entry_spot': trade.entry_spot,
                'exit_spot': trade.exit_spot,
                'status': trade.status,
                'pnl': trade.pnl,
                'volume_5min': trade.volume_5min,
                'sentiment': trade.sentiment,
                'price_change': trade.price_change,
                'entry_reason': trade.entry_reason,
                'exit_reason': trade.exit_reason
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Start timing
    start_time = time.time()
   
    today = date.today()
    today_date = str(today.strftime("%Y-%m-%d"))
    
    # Initialize simulator
    simulator = NiftyOptionsSimulator(
        spot_file='C:\\algotrades\VM\\Chanakya\\SignalGen\\BNF\\summary\\spot_df_'+today_date+'.csv',
        futures_file='C:\\algotrades\VM\\Chanakya\\SignalGen\\BNF\\summary\\nifty_fut_'+today_date+'.csv',
        sentiment_file='C:\\algotrades\VM\\Chanakya\\SignalGen\\BNF\\summary\\cum_sent_df'+today_date+'.csv',
        oi_file='C:\\algotrades\VM\\Chanakya\\SignalGen\\BNF\\summary\\big_oi_summary_rev2'+today_date+'.csv'
    )
    
    # Run simulation
    simulator.run_simulation()
    
    # Get and display results
    results = simulator.get_simulation_results()
    print("\nDetailed Trade Results:")
    
    # Convert results to JSON format with proper indentation
    json_results = results.to_dict(orient='records')
    print(json.dumps(json_results, indent=4, default=str))  # default=str handles datetime objects
    
    # Calculate and display summary statistics
    total_trades = len(results)
    if total_trades > 0:
        winning_trades = len(results[results['pnl'] > 0])
        total_pnl = results['pnl'].sum()
        win_rate = (winning_trades/total_trades)*100
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nSummary Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: {total_pnl:.2f}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        # Plot trades
        simulator.plot_trades()
    else:
        print("\nNo trades were completed during the simulation.") 