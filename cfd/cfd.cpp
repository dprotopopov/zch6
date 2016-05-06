#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

// cfd.cpp: ���������� ����� ����� ��� ����������� ����������.
// ���������� ����������� ����� ���������� ���������
// �������� �������� ����� �������

#include "stdafx.h"
#include <vector>
#include <locale.h>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <assert.h>

// ��������� ������ ��� �������� ������� ������� ������ ���������
typedef struct _data
{
	int x;
	double y;
} data_t;

// ������� ��� ������� ������ ����������� ��������
double target(std::vector<double>& v, std::vector<data_t>& history)
{
	double s = 0.0;
	for (auto it = history.begin(); it != history.end(); ++it)
	{
		double value = v[0] + (v[1] * it->x) + (v[2] / it->x);
		s += (value - it->y) * (value - it->y);
	}
	return s;
}

enum t_ask_mode {
	NOASK = 0,
	ASK = 1
};
enum t_trace_mode {
	NOTRACE = 0,
	TRACE = 1
};
t_ask_mode ask_mode = NOASK;
t_trace_mode trace_mode = NOTRACE;

static const double _e = 1e-8;
static const size_t _n = 3;
static const size_t _m[] = { 10, 10, 10 };

/////////////////////////////////////////////////////////
// ���������� ������� �������� ��������� ������� �� ������ ����
// index - ����� ���� �������
std::vector<unsigned> vector_of(unsigned long index, std::vector<unsigned> m)
{
	std::vector<unsigned> vector;
	for (unsigned i = 0; i < m.size(); i++)
	{
		vector.push_back(index % (1ul + m[i]));
		index /= 1ul + m[i];
	}
	return vector;
}

/////////////////////////////////////////////////////////
// �������������� ������� �������� ��������� �������
// � ������ ��������� �����
// vector - ������ �������� ��������� �������
// m - ����� ��������� �� ������� �� ���������
// a - ������ ����������� ��������� �����
// b - ������ ������������ ��������� �����
std::vector<double> point_of(std::vector<unsigned> vector,
	std::vector<unsigned> m,
	std::vector<double> a,
	std::vector<double> b)
{
	std::vector<double> point = a;
	for (unsigned i = 0; i < m.size(); i++) point[i] += (b[i] - a[i])*vector[i] / m[i];
	return point;
}

/////////////////////////////////////////////////////////
// ���������� ��������� ����� ����� ��������� ���������
double delta(std::vector<double> x, std::vector<double> y)
{
	double diff = 0;
	unsigned i = 0;
	for (; i < x.size() && i < y.size(); i++) diff += (x[i] - y[i])*(x[i] - y[i]);
	for (; i < x.size(); i++) diff += x[i] * x[i];
	for (; i < y.size(); i++) diff += y[i] * y[i];
	return diff;
}

/////////////////////////////////////////////////////////
// ���������� ����� ����� �������
// m - ����� ��������� �� ������� �� ���������
unsigned long total_of(std::vector<unsigned> m)
{
	unsigned long index = 1;
	for (unsigned i = 0; i < m.size(); i++) index *= 1ul + m[i];
	return index;
}

// ��������
double find(std::vector<double> &x,
	std::vector<data_t>& history,
	std::vector<double> a,
	std::vector<double> b,
	std::vector<unsigned> m,
	double e)
{
	for (unsigned i = 0; i < m.size(); i++) assert(m[i]>2);

	unsigned long total = total_of(m);

	while (true)
	{
		// ������� ������ ����� � �������, �������� �������������
		unsigned long index = 0;
		while (index < total)
		{
			x = point_of(vector_of(index++, m), m, a, b);
			break;
		}
		double y = target(x, history);

		while (index < total)
		{
			std::vector<double> x1 = point_of(vector_of(index++, m), m, a, b);
			double y1 = target(x1, history);
			if (y1 >= y) continue;
			x = x1;
			y = y1;
		}

		if (trace_mode == TRACE) for (unsigned i = 0; i < x.size(); i++) std::cout << x[i] << " ";
		if (trace_mode == TRACE) std::cout << "-> " << y << std::endl;

		if (delta(a, b) < e) return y;

		for (unsigned i = 0; i < std::min(a.size(), b.size()); i++) {
			double aa = a[i];
			double bb = b[i];
			double xx = x[i];
			a[i] = std::max(aa, xx - (bb - aa) / m[i]);
			b[i] = std::min(bb, xx + (bb - aa) / m[i]);
		}
	}
}

int main(int argc, char* argv[])
{
	std::vector<data_t> history;
	std::vector<double> a;
	std::vector<double> b;
	std::vector<size_t> m(_m, _m + sizeof(_m) / sizeof(_m[0]));
	size_t n = _n;
	double e = _e;

	char* input_file_name = NULL;
	char* output_file_name = NULL;

	// ��������� ��������� � ������� Windows
	// ������� setlocale() ����� ��� ���������, ������ �������� - ��� ��������� ������, � ����� ������ LC_TYPE - ����� ��������, ������ �������� � �������� ������. 
	// ������ ������� ��������� ����� ������ "Russian", ��� ��������� ������ ������� �������, ����� ����� �������� ����� ����� �� ��� � � ��.
	setlocale(LC_ALL, "");

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-input") == 0) input_file_name = argv[++i];
		else if (strcmp(argv[i], "-output") == 0) output_file_name = argv[++i];
	}

	if (input_file_name != NULL) freopen(input_file_name, "r", stdin);
	if (output_file_name != NULL) freopen(output_file_name, "w", stdout);

	// ��������� �������� �������
	// ������ ������ x y
	std::string line;
	while (std::getline(std::cin, line))
	{
		std::stringstream ss(line);
		data_t data;
		ss >> data.x >> data.y;
		history.push_back(data);
	}

	// ������ ������� �������
	a.push_back(0.0);
	a.push_back(0.0);
	a.push_back(0.0);

	// ������ ������� �������
	double y0 = 0.0;
	double y1 = 0.0;
	double y2 = 0.0;
	for (auto it = history.begin(); it != history.end(); ++it)
	{
		y0 = std::max(y0, it->y);
		y1 = std::max(y1, it->y / it->x);
		y2 = std::max(y2, it->y * it->x);
	}
	b.push_back(y0);
	b.push_back(y1);
	b.push_back(y2);

	std::vector<double> cfd;
	double s = find(cfd, history, a, b, m, e);

	std::cout << "����������� ������������ : " << n << std::endl;
	std::cout << "����� ��������� �������  : "; for (unsigned i = 0; i < m.size(); i++) std::cout << m[i] << " "; std::cout << std::endl;
	std::cout << "����������� ����������   : "; for (unsigned i = 0; i < a.size(); i++) std::cout << a[i] << " "; std::cout << std::endl;
	std::cout << "������������ ����������  : "; for (unsigned i = 0; i < b.size(); i++) std::cout << b[i] << " "; std::cout << std::endl;
	std::cout << "�������� ����������      : " << e << std::endl;
	std::cout << "����� �������� (c,f,d)   : "; for (unsigned i = 0; i < cfd.size(); i++) std::cout << cfd[i] << " "; std::cout << std::endl;
	std::cout << "������ ����� �������     : " << sqrt(cfd[2] / cfd[1]) << std::endl;
	std::cout << "������ ��������          : " << cfd[0]+2*sqrt(cfd[1] * cfd[2]) << std::endl;

	getchar();
	getchar();

	return 0;
}
