"""
Simple Statistics-Based RAG System for Phone Review Analysis
- LLM parses query to extract product + aspects
- Filter data using keyword matching
- Compute statistics on filtered data
- Generate visualizations
- LLM generates final answer
"""

import pandas as pd
import json
import ast
import re
import os
import base64
from typing import List, Dict, Any, Optional
from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


# ============================================================================
# MODULE 1: DATA LOADER
# ============================================================================

class DataLoader:
    """Load and prepare data"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Parse aspects and sentiments
        self.df['aspects'] = self.df['aspects'].apply(self._safe_parse)
        self.df['sentiments'] = self.df['sentiments'].apply(self._safe_parse)
        
        print(f"✅ Loaded {len(self.df)} reviews")
        print(f"📱 Products: {self.df['product'].nunique()}")
        
        return self.df
    
    def _safe_parse(self, x):
        """Parse string to list/dict"""
        if pd.isna(x):
            return [] if '[' in str(x) else {}
        if isinstance(x, (list, dict)):
            return x
        try:
            return ast.literal_eval(x)
        except:
            return [] if '[' in str(x) else {}


# ============================================================================
# MODULE 2: QUERY PARSER (LLM)
# ============================================================================

class QueryParser:
    """Use LLM to parse user query"""
    
    def __init__(self, client: OpenAI, available_products: List[str]):
        self.client = client
        self.available_products = available_products
    
    def parse_query(self, query: str) -> Dict:
        """Parse query to extract intent, products, aspects"""
        
        system_prompt = f"""Bạn là trợ lý phân tích câu hỏi về đánh giá điện thoại.

Nhiệm vụ: Phân tích câu hỏi để trích xuất:
1. products: Danh sách tên sản phẩm (có thể viết tắt/không chính xác)
2. aspects: Danh sách khía cạnh được hỏi (general, battery, camera, performance, screen, design, price, storage, features, ser&acc)
3. sentiment_focus: Người dùng muốn biết ưu điểm (positive), nhược điểm (negative), hay tổng quan (null)
4. is_comparison: Có phải câu hỏi so sánh không? (true/false)

ASPECTS có thể có:
- general: đánh giá chung
- battery/pin: pin
- camera: camera
- performance: hiệu năng, tốc độ, chip
- screen: màn hình
- design: thiết kế, ngoại hình
- price: giá cả
- storage: bộ nhớ, lưu trữ
- features: tính năng
- ser&acc: dịch vụ, phụ kiện

Danh sách sản phẩm có sẵn (để tham khảo):
{', '.join(self.available_products[:10])}...

Trả về JSON với format:
{{
  "products": ["product_name1", "product_name2"],
  "aspects": ["aspect1", "aspect2"],
  "sentiment_focus": "positive|negative|null",
  "is_comparison": true|false
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Câu hỏi: {query}\n\nPhân tích:"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            parsed = json.loads(response.choices[0].message.content)
            print(f"🔍 Parsed: {parsed}")
            return parsed
            
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return {
                "products": [],
                "aspects": ["general"],
                "sentiment_focus": None,
                "is_comparison": False
            }


# ============================================================================
# MODULE 3: DATA FILTER
# ============================================================================

class DataFilter:
    """Filter data based on parsed query"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.all_products = df['product'].unique().tolist()
    
    def keyword_match_product(self, query_product: str) -> List[str]:
        """Match products using keyword extraction with exact priority"""
        query_lower = query_product.lower().strip()
        
        # Extract keywords from query (split and clean)
        keywords = []
        for word in query_lower.split():
            # Remove common words and keep meaningful parts
            if len(word) >= 2 and word not in ['điện', 'thoại', 'máy', 'chiếc', 'cái', 'gb', 'tb']:
                keywords.append(word)
        
        if not keywords:
            return []
        
        exact_matches = []
        partial_matches = []
        
        for product in self.all_products:
            product_lower = product.lower()
            
            # Strategy 1: Exact phrase match (highest priority)
            if query_lower in product_lower:
                pattern = r'\b' + re.escape(query_lower) + r'\b'
                if re.search(pattern, product_lower):
                    exact_matches.append(product)
                    continue
            
            # Strategy 2: All keywords must exist (partial match)
            if all(kw in product_lower for kw in keywords):
                product_words = set(re.findall(r'\b\w+\b', product_lower))
                query_words = set(keywords)
                
                significant_extras = ['pro', 'max', 'plus', 'ultra', 'mini', 'lite', 'note']
                extra_in_product = product_words.intersection(significant_extras)
                extra_in_query = query_words.intersection(significant_extras)
                
                if extra_in_product - extra_in_query:
                    continue
                
                partial_matches.append(product)
        
        if exact_matches:
            return exact_matches
        return partial_matches
    
    def filter_data(self, products: List[str], aspects: List[str]) -> tuple:
        """Filter dataframe based on products and aspects"""
        
        matched_products = []
        
        if not products:
            matched_products = self.all_products
        else:
            for query_prod in products:
                matched = self.keyword_match_product(query_prod)
                matched_products.extend(matched)
            
            matched_products = list(set(matched_products))
        
        if not matched_products:
            print("⚠️ No products matched, using all products")
            matched_products = self.all_products
        else:
            print(f"✅ Matched {len(matched_products)} products: {matched_products[:3]}{'...' if len(matched_products) > 3 else ''}")
        
        product_filtered = self.df[self.df['product'].isin(matched_products)].copy()
        
        product_review_counts = {}
        for product in matched_products:
            product_df = product_filtered[product_filtered['product'] == product]
            if 'review_id' in product_df.columns:
                product_review_counts[product] = product_df['review_id'].nunique()
            else:
                product_review_counts[product] = product_df['review'].nunique() if 'review' in product_df.columns else len(product_df)
        
        if aspects and aspects != ['general']:
            def has_aspect_match(review_aspects):
                if not isinstance(review_aspects, list):
                    return False
                return any(asp in aspects for asp in review_aspects)
            
            aspect_filtered = product_filtered[product_filtered['aspects'].apply(has_aspect_match)].copy()
        else:
            aspect_filtered = product_filtered.copy()
        
        print(f"📊 Filtered to {len(aspect_filtered)} review entries")
        
        return aspect_filtered, matched_products, product_review_counts


# ============================================================================
# MODULE 4: STATISTICS COMPUTER
# ============================================================================

class StatisticsComputer:
    """Compute statistics on filtered data"""
    
    @staticmethod
    def compute_stats(df: pd.DataFrame, aspects: List[str], product_review_counts: Dict[str, int]) -> Dict:
        """Compute comprehensive statistics"""
        
        if len(df) == 0:
            return {
                'total_reviews': 0,
                'message': 'Không tìm thấy dữ liệu phù hợp'
            }
        
        total_reviews = sum(product_review_counts.values())
        
        stats = {
            'total_reviews': total_reviews,
            'products': {},
            'aspects': {},
            'overall_sentiment': {}
        }
        
        all_sentiments = []
        for sentiments_dict in df['sentiments']:
            if isinstance(sentiments_dict, dict):
                all_sentiments.extend(sentiments_dict.values())
        
        sentiment_counts = Counter(all_sentiments)
        
        stats['overall_sentiment'] = {
            'positive': sentiment_counts.get('Positive', 0),
            'neutral': sentiment_counts.get('Neutral', 0),
            'negative': sentiment_counts.get('Negative', 0),
            'positive_pct': sentiment_counts.get('Positive', 0) / total_reviews * 100 if total_reviews > 0 else 0,
            'negative_pct': sentiment_counts.get('Negative', 0) / total_reviews * 100 if total_reviews > 0 else 0,
            'neutral_pct': sentiment_counts.get('Neutral', 0) / total_reviews * 100 if total_reviews > 0 else 0
        }
        
        for product, product_total_reviews in product_review_counts.items():
            product_df = df[df['product'] == product]
            stats['products'][product] = StatisticsComputer._compute_product_stats(
                product_df, 
                product_total_reviews
            )
        
        for aspect in aspects:
            aspect_reviews = []
            aspect_sentiments = []
            
            for idx, row in df.iterrows():
                if isinstance(row['aspects'], list) and aspect in row['aspects']:
                    aspect_reviews.append(row['sentence'])
                    if isinstance(row['sentiments'], dict) and aspect in row['sentiments']:
                        aspect_sentiments.append(row['sentiments'][aspect])
            
            sentiment_counts = Counter(aspect_sentiments)
            positive_count = sentiment_counts.get('Positive', 0)
            negative_count = sentiment_counts.get('Negative', 0)
            neutral_count = sentiment_counts.get('Neutral', 0)
            
            implicit_neutral = total_reviews - len(aspect_reviews)
            total_neutral = neutral_count + implicit_neutral
            
            stats['aspects'][aspect] = {
                'count': len(aspect_reviews),
                'positive': positive_count,
                'neutral': total_neutral,
                'negative': negative_count,
                'positive_pct': positive_count / total_reviews * 100 if total_reviews > 0 else 0,
                'negative_pct': negative_count / total_reviews * 100 if total_reviews > 0 else 0,
                'neutral_pct': total_neutral / total_reviews * 100 if total_reviews > 0 else 0,
                'mentioned_pct': len(aspect_reviews) / total_reviews * 100 if total_reviews > 0 else 0,
                'sample_reviews': aspect_reviews[:3]
            }
        
        return stats
    
    @staticmethod
    def _compute_product_stats(product_df: pd.DataFrame, total_reviews: int) -> Dict:
        """Compute stats for a single product"""
        all_sentiments = []
        for sentiments_dict in product_df['sentiments']:
            if isinstance(sentiments_dict, dict):
                all_sentiments.extend(sentiments_dict.values())
        
        sentiment_counts = Counter(all_sentiments)
        
        return {
            'review_count': total_reviews,
            'positive': sentiment_counts.get('Positive', 0),
            'neutral': sentiment_counts.get('Neutral', 0),
            'negative': sentiment_counts.get('Negative', 0),
            'positive_pct': sentiment_counts.get('Positive', 0) / total_reviews * 100 if total_reviews > 0 else 0,
            'negative_pct': sentiment_counts.get('Negative', 0) / total_reviews * 100 if total_reviews > 0 else 0,
            'neutral_pct': sentiment_counts.get('Neutral', 0) / total_reviews * 100 if total_reviews > 0 else 0
        }


# ============================================================================
# MODULE 5: VISUALIZATION
# ============================================================================

class ChartGenerator:
    """Generate visualization charts from statistics"""
    
    @staticmethod
    def create_sentiment_pie(stats: Dict, title: str = "Phân bố cảm xúc") -> str:
        """Create pie chart for sentiment distribution"""
        if 'overall_sentiment' not in stats:
            return None
        
        s = stats['overall_sentiment']
        labels = ['Tích cực', 'Trung lập', 'Tiêu cực']
        sizes = [s['positive'], s['neutral'], s['negative']]
        colors = ['#4CAF50', '#FFC107', '#F44336']
        explode = (0.05, 0, 0.05)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode,
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_base64
    
    @staticmethod
    def create_product_comparison(stats: Dict) -> str:
        """Create bar chart comparing products"""
        if 'products' not in stats or len(stats['products']) < 2:
            return None
        
        products = []
        positive = []
        negative = []
        
        for product, pstats in stats['products'].items():
            products.append(product.split()[0:3])
            positive.append(pstats['positive_pct'])
            negative.append(pstats['negative_pct'])
        
        products = [' '.join(p) for p in products]
        
        x = range(len(products))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar([i - width/2 for i in x], positive, width, 
                       label='Tích cực', color='#4CAF50', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], negative, width,
                       label='Tiêu cực', color='#F44336', alpha=0.8)
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Sản phẩm', fontsize=12, weight='bold')
        ax.set_ylabel('Phần trăm (%)', fontsize=12, weight='bold')
        ax.set_title('So sánh đánh giá giữa các sản phẩm', fontsize=14, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(products, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_base64
    
    @staticmethod
    def create_aspect_breakdown(stats: Dict) -> str:
        """Create horizontal bar chart for aspect breakdown"""
        if 'aspects' not in stats or not stats['aspects']:
            return None
        
        aspects = []
        positive = []
        negative = []
        mentioned = []
        
        for aspect, astats in stats['aspects'].items():
            aspects.append(aspect.upper())
            positive.append(astats['positive_pct'])
            negative.append(astats['negative_pct'])
            mentioned.append(astats['mentioned_pct'])
        
        sorted_data = sorted(zip(aspects, positive, negative, mentioned), 
                           key=lambda x: x[3], reverse=True)
        aspects, positive, negative, mentioned = zip(*sorted_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        y_pos = range(len(aspects))
        
        ax1.barh(y_pos, positive, color='#4CAF50', alpha=0.8, label='Tích cực')
        ax1.barh(y_pos, [-n for n in negative], color='#F44336', alpha=0.8, label='Tiêu cực')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(aspects)
        ax1.set_xlabel('Phần trăm (%)', fontsize=11, weight='bold')
        ax1.set_title('Cảm xúc theo khía cạnh', fontsize=13, weight='bold')
        ax1.legend(loc='lower right')
        ax1.axvline(x=0, color='black', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
        
        bars = ax2.barh(y_pos, mentioned, color='#2196F3', alpha=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, mentioned)):
            ax2.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(aspects)
        ax2.set_xlabel('Phần trăm được đề cập (%)', fontsize=11, weight='bold')
        ax2.set_title('Tỷ lệ đề cập khía cạnh', fontsize=13, weight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_base64
    
    @staticmethod
    def generate_all_charts(stats: Dict, parsed_query: Dict) -> Dict[str, str]:
        """Generate all relevant charts based on query type"""
        charts = {}
        
        pie_chart = ChartGenerator.create_sentiment_pie(stats)
        if pie_chart:
            charts['sentiment_pie'] = pie_chart
        
        if len(stats.get('products', {})) > 1 or parsed_query.get('is_comparison'):
            comparison_chart = ChartGenerator.create_product_comparison(stats)
            if comparison_chart:
                charts['product_comparison'] = comparison_chart
        
        if stats.get('aspects') and len(stats['aspects']) > 0:
            aspect_chart = ChartGenerator.create_aspect_breakdown(stats)
            if aspect_chart:
                charts['aspect_breakdown'] = aspect_chart
        
        return charts


# ============================================================================
# MODULE 6: CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """Build context for LLM from statistics"""
    
    @staticmethod
    def build_context(stats: Dict, df: pd.DataFrame, parsed_query: Dict) -> str:
        """Build comprehensive context"""
        
        if stats.get('total_reviews', 0) == 0:
            return "Không tìm thấy dữ liệu đánh giá phù hợp với câu hỏi."
        
        parts = []
        
        parts.append(f"=== TỔNG QUAN ===")
        parts.append(f"Tổng số đánh giá: {stats['total_reviews']}")
        
        if 'overall_sentiment' in stats:
            s = stats['overall_sentiment']
            parts.append(f"\nPhân bố cảm xúc tổng thể:")
            parts.append(f"  • Tích cực: {s['positive_pct']:.1f}% ({s['positive']} đánh giá)")
            parts.append(f"  • Trung lập: {s['neutral_pct']:.1f}% ({s['neutral']} đánh giá)")
            parts.append(f"  • Tiêu cực: {s['negative_pct']:.1f}% ({s['negative']} đánh giá)")
        
        if len(stats['products']) > 1 or parsed_query.get('is_comparison'):
            parts.append(f"\n=== SO SÁNH SẢN PHẨM ===")
            for product, pstats in stats['products'].items():
                parts.append(f"\n{product}:")
                parts.append(f"  • Số đánh giá: {pstats['review_count']}")
                parts.append(f"  • Tích cực: {pstats['positive_pct']:.1f}% ({pstats['positive']})")
                parts.append(f"  • Tiêu cực: {pstats['negative_pct']:.1f}% ({pstats['negative']})")
        
        if stats['aspects']:
            parts.append(f"\n=== PHÂN TÍCH THEO KHÍA CẠNH ===")
            for aspect, astats in stats['aspects'].items():
                parts.append(f"\n{aspect.upper()}:")
                parts.append(f"  • Số đánh giá đề cập: {astats['count']}/{stats['total_reviews']} ({astats['mentioned_pct']:.1f}%)")
                parts.append(f"  • Tích cực: {astats['positive_pct']:.1f}% ({astats['positive']} đánh giá)")
                parts.append(f"  • Tiêu cực: {astats['negative_pct']:.1f}% ({astats['negative']} đánh giá)")
                parts.append(f"  • Trung lập/Không đề cập: {astats['neutral_pct']:.1f}% ({astats['neutral']} đánh giá)")
                
                if astats['sample_reviews']:
                    parts.append(f"  • Ví dụ:")
                    for i, review in enumerate(astats['sample_reviews'][:2], 1):
                        parts.append(f"    {i}. {review[:150]}")
        
        parts.append(f"\n=== MẪU ĐÁNH GIÁ ===")
        sample_df = df.head(5)
        for i, row in enumerate(sample_df.iterrows(), 1):
            idx, r = row
            parts.append(f"\n{i}. [{r['product']}]")
            parts.append(f"   Khía cạnh: {', '.join(r['aspects']) if isinstance(r['aspects'], list) else 'N/A'}")
            parts.append(f"   Nội dung: {r['sentence'][:200]}")
        
        return "\n".join(parts)


# ============================================================================
# MODULE 7: RAG SYSTEM
# ============================================================================

class SimpleRAG:
    """Main RAG system"""
    
    def __init__(self, csv_path: str, openai_api_key: Optional[str] = None):
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OpenAI API key not found! Add to .env file or pass as parameter")
        
        self.client = OpenAI(api_key=api_key)
        
        loader = DataLoader(csv_path)
        self.df = loader.load_data()
        
        self.parser = QueryParser(self.client, self.df['product'].unique().tolist())
        self.filter = DataFilter(self.df)
        
        print("✅ System ready!")
    
    def answer(self, query: str, show_charts: bool = True) -> Dict[str, Any]:
        """Main method: answer user query
        
        Args:
            query: User question
            show_charts: Whether to generate visualization charts
            
        Returns:
            Dict with 'answer', 'charts' (base64 encoded images), and 'stats'
        """
        
        print(f"\n{'='*80}")
        print(f"❓ Query: {query}")
        print(f"{'='*80}")
        
        # Step 1: Parse query with LLM
        print("\n🔍 Step 1: Parsing query...")
        parsed = self.parser.parse_query(query)
        
        # Step 2: Filter data
        print("\n📊 Step 2: Filtering data...")
        filtered_df, matched_products, product_review_counts = self.filter.filter_data(
            parsed.get('products', []),
            parsed.get('aspects', ['general'])
        )
        
        if len(filtered_df) == 0:
            return {
                'answer': "⚠️ Không tìm thấy dữ liệu phù hợp với câu hỏi của bạn.",
                'charts': {},
                'stats': {}
            }
        
        # Step 3: Compute statistics
        print("\n📈 Step 3: Computing statistics...")
        stats = StatisticsComputer.compute_stats(
            filtered_df,
            parsed.get('aspects', ['general']),
            product_review_counts
        )
        
        # Step 4: Build context
        print("\n📝 Step 4: Building context...")
        context = ContextBuilder.build_context(stats, filtered_df, parsed)
        
        # Step 5: Generate charts
        charts = {}
        if show_charts:
            print("\n📊 Step 5: Generating charts...")
            charts = ChartGenerator.generate_all_charts(stats, parsed)
            print(f"✅ Generated {len(charts)} chart(s)")
        
        # Step 6: Generate answer with LLM
        print("\n🤖 Step 6: Generating answer...")
        answer = self._generate_answer(query, context, parsed)
        
        return {
            'answer': answer,
            'charts': charts,
            'stats': stats
        }
    
    def _generate_answer(self, query: str, context: str, parsed_query: Dict) -> str:
        """Generate final answer using LLM"""
        
        system_prompt = """Bạn là trợ lý phân tích đánh giá điện thoại chuyên nghiệp.

NHIỆM VỤ:
Dựa trên dữ liệu thống kê được cung cấp, hãy trả lời câu hỏi của người dùng một cách:
- Chính xác, dựa trên số liệu cụ thể
- Cân bằng, không thiên vị
- Dễ hiểu, súc tích
- Trích dẫn % và số lượng review

CẤU TRÚC TRẢ LỜI:
1. Tóm tắt ngắn gọn (1-2 câu)
2. Phân tích số liệu chi tiết
3. Đưa ra ví dụ từ review (nếu có)
4. Kết luận

LƯU Ý:
- KHÔNG bịa đặt thông tin ngoài context
- Nếu dữ liệu không đủ, nói rõ hạn chế
- Ưu tiên số liệu thống kê hơn review đơn lẻ"""

        user_prompt = f"""Dữ liệu thống kê:
{context}

Câu hỏi: {query}

Hãy trả lời dựa trên dữ liệu trên:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            usage = response.usage
            print(f"💰 Tokens: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})")
            
            return answer
            
        except Exception as e:
            return f"⚠️ Lỗi khi generate answer: {e}\n\nContext:\n{context}"
    
    def get_available_products(self) -> List[str]:
        """Get list of available products"""
        return self.df['product'].unique().tolist()
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset overview statistics"""
        return {
            'total_reviews': len(self.df),
            'total_products': self.df['product'].nunique(),
            'products': self.df['product'].value_counts().head(10).to_dict()
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the system"""
    
    CSV_PATH = "clean_reviews.csv"
    
    print("="*80)
    print("🚀 SIMPLE STATISTICS RAG SYSTEM WITH VISUALIZATION")
    print("="*80)
    
    rag = SimpleRAG(CSV_PATH)
    
    print("\n" + "="*80)
    print("✅ READY TO USE!")
    print("="*80)
    print("\n📝 Example usage:")
    print("result = rag.answer('Pin Xiaomi 15T có tốt không?')")
    print("print(result['answer'])")
    
    return rag


def demo():
    """Interactive demo with chart display"""
    rag = main()
    
    print("\n" + "="*80)
    print("🎮 INTERACTIVE DEMO")
    print("="*80)
    print("Type 'exit' to quit")
    print("-"*80)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = rag.answer(user_input)
            
            print(f"\n🤖 Answer:\n{result['answer']}\n")
            
            if result['charts']:
                print(f"📊 Generated {len(result['charts'])} chart(s):")
                for chart_name, img_base64 in result['charts'].items():
                    filename = f"{chart_name}.png"
                    with open(filename, 'wb') as f:
                        f.write(base64.b64decode(img_base64))
                    print(f"  ✅ Saved: {filename}")
                print(f"\n💡 Tip: Open the PNG files to view charts!\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")


if __name__ == "__main__":
    demo()